"""Byte-level streaming dataset backed by mmap .bin files.

Pack text corpora with `prepare_data.py`, then point ByteFlowBinStreamer at the
.bin path.  The streamer memory-maps the file and reads random byte sequences
— zero RAM, perfect global shuffle, no tokenizer overhead.
"""
import os
import torch
import numpy as np
from torch.utils.data import IterableDataset


class ByteFlowBinStreamer(IterableDataset):
    """Stream real bytes from a memory-mapped .bin with ideal global shuffle.

    Falls back to synthetic random data when the .bin is absent (tests / CI).
    Batches are homogeneous: all-text or all-image (controlled by image_prob).
    """

    PAD = 268
    IMG_DIM = 768

    def __init__(self, bin_path=None, batch_size=4, seq_len=64, image_prob=0.0,
                 max_patch_len=16):
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
            patches = torch.full((B, T, self.IMG_DIM), float(self.PAD))
            lengths = torch.zeros(B, T, dtype=torch.long)
            is_img = torch.zeros(B, T, dtype=torch.bool)
            for b in range(B):
                for t in range(T):
                    ll = torch.randint(1, L + 1, (1,)).item()
                    lengths[b, t] = ll
                    patches[b, t, :ll] = torch.randint(0, 268, (ll,)).float()
            return patches, lengths, is_img

        # --- real mmap path -------------------------------------------
        is_image = torch.rand(1).item() < self.image_prob
        if is_image:
            is_img = torch.ones(B, T, dtype=torch.bool)
            patches = torch.rand(B, T, self.IMG_DIM)
            lengths = torch.zeros(B, T, dtype=torch.long)
        else:
            is_img = torch.zeros(B, T, dtype=torch.bool)
            patches = torch.full((B, T, self.IMG_DIM), float(self.PAD))
            lengths = torch.zeros(B, T, dtype=torch.long)

            need = T * L
            max_off = max(1, self.data_size - need - 1)
            for b in range(B):
                start = np.random.randint(0, max_off)
                chunk = self.data[start: start + need].copy()
                rows = torch.as_tensor(chunk, dtype=torch.float32).view(T, L)
                patches[b, :, :L] = rows
                lengths[b, :] = (rows != self.PAD).sum(dim=-1)

        return patches, lengths, is_img

    def __iter__(self):
        while True:
            yield self._make_batch()


def create_loader(batch_size=4, seq_len=64, image_prob=0.5, data_path=None):
    """Create a data loader.  Falls back to synthetic data when data_path is None."""
    return ByteFlowBinStreamer(data_path, batch_size, seq_len, image_prob)
