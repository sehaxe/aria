"""Byte-level streaming dataset.

Two backends:
- ByteFlowBinStreamer: mmap .bin (zero RAM, global shuffle via random offset).
- TxtStreamer: read .txt files directly from a folder (no packing needed),
  global shuffle by shuffling file order + lines. Optional (bin not required).
"""
import os
import glob
import json
import random
import time
import collections
import threading
import torch
import numpy as np
from torch.utils.data import IterableDataset

# parquet (wiki/books/qa come as HF parquet) — lazy import, optional dep
try:
    import pyarrow.parquet as pq
except Exception:
    pq = None

# text column names per dataset schema (HF parquet layouts)
_PARQUET_TEXT_COLS = {
    "wikipedia": ["text"],
    "fineweb-edu": ["text"],
    "natural_questions": ["question", "answer"],
}

def _parquet_text(fp):
    """Extract text documents from a HF parquet file."""
    if pq is None:
        return []
    try:
        t = pq.read_table(fp)
        cols = set(t.column_names)
        # pick schema by available columns
        if "text" in cols:
            key = "text"
        elif "question" in cols and "answer" in cols:
            key = "qa"
        else:
            # fallback: first string-ish column
            key = t.column_names[0]
        out = []
        if key == "qa":
            q = t.column("question").to_pylist()
            a = t.column("answer").to_pylist()
            for qq, aa in zip(q, a):
                if qq:
                    out.append(f"Q: {qq}\nA: {aa or ''}")
        else:
            for v in t.column(key).to_pylist():
                if v and isinstance(v, str):
                    out.append(v)
        return out
    except Exception:
        return []


class ByteFlowBinStreamer(IterableDataset):
    """Stream real bytes from a memory-mapped .bin with ideal global shuffle.

    Falls back to synthetic random data when the .bin is absent (tests / CI).
    Batches are homogeneous: all-text or all-image (controlled by image_prob).
    """

    PAD = 268
    IMG_DIM = 768

    def __init__(self, bin_path=None, batch_size=4, seq_len=64, image_prob=0.0,
                 max_patch_len=128):
        self.bin_path = bin_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.image_prob = image_prob
        self.max_patch_len = max_patch_len

        if bin_path is None or not os.path.exists(bin_path):
            print(f"[Dataset] {bin_path} not found — synthetic fallback.")
            self.data = None
            self.data_size = 0
        else:
            self.data = np.memmap(bin_path, dtype=np.uint8, mode="r")
            self.data_size = len(self.data)
            print(f"[Dataset] mmap {bin_path}  ({self.data_size:,} bytes)")

    def _make_batch(self):
        B, T, L = self.batch_size, self.seq_len, self.max_patch_len

        # --- synthetic fallback (no .bin available) -------------------
        if self.data is None:
            patches = torch.full((B, T, self.IMG_DIM), self.PAD, dtype=torch.int32)
            lengths = torch.zeros(B, T, dtype=torch.long)
            is_img = torch.zeros(B, T, dtype=torch.bool)
            for b in range(B):
                for t in range(T):
                    ll = int(torch.randint(1, L + 1, (1,)).item())
                    lengths[b, t] = ll
                    patches[b, t, :ll] = torch.randint(0, 268, (ll,), dtype=torch.int32)
            return patches, lengths, is_img

        # --- real mmap path -------------------------------------------
        # PAD_ID=268 exceeds uint8, so store as int32 (half the bytes of the
        # original float32, and .long() on-GPU is free). Avoids the 4x
        # CPU float32 materialization per batch.
        is_image = torch.rand(1).item() < self.image_prob
        if is_image:
            is_img = torch.ones(B, T, dtype=torch.bool)
            patches = torch.randint(0, 256, (B, T, self.IMG_DIM), dtype=torch.int32)
            lengths = torch.zeros(B, T, dtype=torch.long)
        else:
            is_img = torch.zeros(B, T, dtype=torch.bool)
            patches = torch.full((B, T, self.IMG_DIM), self.PAD, dtype=torch.int32)
            lengths = torch.zeros(B, T, dtype=torch.long)

            need = T * L
            max_off = max(1, self.data_size - need - 1)
            for b in range(B):
                start = int(np.random.randint(0, max_off))
                # mmap slice -> numpy uint8 (cheap copy), then int32. No float32.
                chunk = np.array(self.data[start: start + need], dtype=np.uint8)
                chunk = torch.from_numpy(chunk.astype(np.int32, copy=False)).view(T, L)
                patches[b, :, :L] = chunk
                # vectorized length: count non-PAD bytes up to first PAD
                pad_mask = (chunk != self.PAD)
                lengths[b, :] = pad_mask.long().sum(dim=-1)

        return patches, lengths, is_img

    def __iter__(self):
        while True:
            yield self._make_batch()


class TxtStreamer(IterableDataset):
    """Stream bytes directly from a folder of .txt files (no .bin packing).

    Category-balanced global shuffle: files are grouped by subfolder
    (mix/<cat>/), each category is sampled proportionally to CAT_BUDGET.
    Lines within a chosen file are shuffled. Reads lazily (one file at a
    time) -> zero RAM regardless of corpus size. image_prob ignored.
    """

    PAD = 268
    IMG_DIM = 768

    # CAT_BUDGET weights (mirrors scripts/build_corpus.py)
    CAT_BUDGET = {
        "code": 0.29, "agentic": 0.22, "math": 0.11, "web": 0.13, "ruweb": 0.05,
        "wiki": 0.10, "books": 0.05, "reasoning": 0.04, "synthetic": 0.005,
        "qa": 0.005,
    }

    def __init__(self, folder, batch_size=4, seq_len=64, image_prob=0.0,
                 max_patch_len=128, seed=1337):
        self.folder = folder
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.image_prob = image_prob
        self.max_patch_len = max_patch_len
        self.rng = random.Random(seed)
        # group files by category subfolder
        self.cat_files = {}
        for cat in self.CAT_BUDGET:
            d = os.path.join(folder, cat)
            if os.path.isdir(d):
                fs = sorted(glob.glob(os.path.join(d, "*.txt"))) + \
                     sorted(glob.glob(os.path.join(d, "*.jsonl"))) + \
                     sorted(glob.glob(os.path.join(d, "*.parquet")))
                if fs:
                    self.cat_files[cat] = fs
        if not self.cat_files:
            print(f"[Dataset] no category folders in {folder} — synthetic fallback.")
        else:
            total = sum(len(v) for v in self.cat_files.values())
            cats = list(self.cat_files.keys())
            w = [self.CAT_BUDGET.get(c, 0.0) for c in cats]
            s = sum(w) or 1.0
            self.cats = cats
            self.weights = [x / s for x in w]
            print(f"[Dataset] txt-stream {folder}: {total} files across {len(cats)} cats")

    def _read_docs(self, fp):
        """Yield text documents from a file. .txt -> lines; .jsonl -> conversations; .parquet -> HF text cols."""
        if fp.endswith(".parquet"):
            for d in _parquet_text(fp):
                yield d
            return
        if fp.endswith(".jsonl"):
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            r = json.loads(line)
                        except Exception:
                            continue
                        conv = r.get("conversations") or r.get("conversation") or []
                        parts = []
                        for m in conv:
                            role = m.get("from") or m.get("role") or "?"
                            val = m.get("value") or m.get("text") or ""
                            parts.append(f"{role}: {val}")
                        if parts:
                            yield "\n\n".join(parts)
            except Exception:
                return
        else:
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f.read().split("\n"):
                        line = line.strip()
                        if line:
                            yield line
            except Exception:
                return

    def _sample_line(self):
        cat = self.rng.choices(self.cats, weights=self.weights, k=1)[0]
        fp = self.rng.choice(self.cat_files[cat])
        docs = list(self._read_docs(fp))
        if not docs:
            return None
        self.rng.shuffle(docs)
        return docs[0]

    def _make_batch(self):
        B, T, L = self.batch_size, self.seq_len, self.max_patch_len
        if not self.cat_files:
            patches = torch.full((B, T, self.IMG_DIM), float(self.PAD))
            lengths = torch.zeros(B, T, dtype=torch.long)
            is_img = torch.zeros(B, T, dtype=torch.bool)
            for b in range(B):
                for t in range(T):
                    ll = random.randint(1, L)
                    lengths[b, t] = ll
                    patches[b, t, :ll] = torch.randint(0, 268, (ll,)).float()
            return patches, lengths, is_img

        is_img = torch.zeros(B, T, dtype=torch.bool)
        patches = torch.full((B, T, self.IMG_DIM), float(self.PAD))
        lengths = torch.zeros(B, T, dtype=torch.long)
        for b in range(B):
            for t in range(T):
                line = self._sample_line()
                if not line:
                    continue
                toks = [min(ord(c), 267) for c in line[:L]]
                if not toks:
                    continue
                patches[b, t, :len(toks)] = torch.tensor(toks, dtype=torch.float32)
                lengths[b, t] = len(toks)
        return patches, lengths, is_img

    def __iter__(self):
        while True:
            yield self._make_batch()


def create_loader(batch_size=4, seq_len=64, image_prob=0.5, data_path=None,
                  prefetch=2):
    """Create a data loader.

    - data_path ends with .bin  -> ByteFlowBinStreamer (mmap, global shuffle)
    - data_path is a folder     -> TxtStreamer (direct .txt, global shuffle)
    - None                      -> synthetic fallback

    Wrapped in a single-process PrefetchLoader: a background thread prepares
    the next batch(es) while the GPU computes the current one, so CPU data
    prep (mmap slice, uint8->tensor) never blocks the accelerator. Multiprocess
    DataLoader is intentionally avoided -- free-threaded CPython 3.15t deadlocks
    its resource_tracker under num_workers>0.
    """
    if data_path and os.path.isdir(data_path):
        ds = TxtStreamer(data_path, batch_size, seq_len, image_prob)
    else:
        ds = ByteFlowBinStreamer(data_path, batch_size, seq_len, image_prob)
    return PrefetchLoader(ds, prefetch=prefetch)


class PrefetchLoader:
    """Overlap data preparation with GPU compute via a single background thread.

    The dataset yields (patches, lengths, is_img[, targets]) tuples of tensors;
    this wrapper pre-materializes `prefetch` of them ahead of time so
    train_phased's `next(loader_iter)` returns instantly.
    """

    def __init__(self, dataset, prefetch=2):
        self.dataset = dataset
        self.prefetch = max(1, prefetch)
        self._iter = iter(dataset)
        self._queue = collections.deque()
        self._thread = None
        self._stop = False
        self._fill()

    def _fill(self):
        def _producer():
            while not self._stop:
                try:
                    item = next(self._iter)
                except StopIteration:
                    break
                self._queue.append(item)
                # brief yield so the consumer can drain
                time.sleep(0)

        self._thread = threading.Thread(target=_producer, daemon=True)
        self._thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        # Wait until the producer has at least one item ready.
        while not self._queue and not self._stop:
            time.sleep(0.001)
        if not self._queue:
            raise StopIteration
        return self._queue.popleft()

    def __del__(self):
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=1.0)
