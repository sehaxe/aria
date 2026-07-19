"""Pack the mixed corpus (txt/jsonl/parquet) into ONE .bin for mmap streaming.

Why: TxtStreamer does f.read() on whole files every sample -> kills throughput
on the slow external HDD (3 min/step). ByteFlowBinStreamer reads via mmap
(lazy page-cache), which is ~100x faster. The Aria model is byte-level
(256-vocab), so we just concatenate raw document bytes with a doc separator.

Category balance (CAT_BUDGET) is preserved by weighted interleaving: we draw
documents from each category in proportion to its budget weight while writing.
"""
import os, sys, glob, json, random
from pathlib import Path

MIX = "/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain/mix"
OUT = "/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain/corpus.bin"
SEED = 1337

CAT_BUDGET = {
    "code": 0.29, "agentic": 0.22, "math": 0.11, "web": 0.13, "ruweb": 0.05,
    "wiki": 0.10, "books": 0.05, "reasoning": 0.04, "synthetic": 0.005, "qa": 0.005,
}
SEP = b"\n\n"  # document boundary


def iter_txt(fp):
    try:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f.read().split("\n"):
                line = line.strip()
                if line:
                    yield line
    except Exception:
        return


def iter_jsonl(fp):
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


def iter_parquet(fp):
    try:
        import pyarrow.parquet as pq
    except Exception:
        return
    try:
        t = pq.read_table(fp)
        cols = set(t.column_names)
        if "text" in cols:
            for v in t.column("text").to_pylist():
                if v and isinstance(v, str):
                    yield v
        elif "question" in cols and "answer" in cols:
            q = t.column("question").to_pylist()
            a = t.column("answer").to_pylist()
            for qq, aa in zip(q, a):
                if qq:
                    yield f"Q: {qq}\nA: {aa or ''}"
    except Exception:
        return


def cat_docs(cat_dir):
    """Yield all document strings from a category folder, streaming file by file."""
    txt = sorted(glob.glob(os.path.join(cat_dir, "*.txt")))
    jsonl = sorted(glob.glob(os.path.join(cat_dir, "*.jsonl")))
    parquet = sorted(glob.glob(os.path.join(cat_dir, "*.parquet")))
    for fp in txt:
        for d in iter_txt(fp):
            yield d
    for fp in jsonl:
        for d in iter_jsonl(fp):
            yield d
    for fp in parquet:
        for d in iter_parquet(fp):
            yield d


def main():
    rng = random.Random(SEED)
    cats = [c for c in CAT_BUDGET if os.path.isdir(os.path.join(MIX, c))]
    weights = [CAT_BUDGET[c] for c in cats]
    print(f"Packing cats: {cats}")
    # open one streaming generator per category
    gens = {c: cat_docs(os.path.join(MIX, c)) for c in cats}
    total = 0
    with open(OUT, "wb") as out:
        # weighted round-robin: each draw picks a category by budget weight
        i = 0
        while True:
            cat = rng.choices(cats, weights=weights, k=1)[0]
            try:
                doc = next(gens[cat])
            except StopIteration:
                # exhausted this category's generator; drop it
                del gens[cat]
                ci = cats.index(cat)
                cats.pop(ci); weights.pop(ci)
                if not cats:
                    break
                continue
            if not doc:
                continue
            out.write(doc.encode("utf-8", "ignore"))
            out.write(SEP)
            total += 1
            i += 1
            if i % 100000 == 0:
                print(f"  {i:,} docs written", flush=True)
    gb = total and os.path.getsize(OUT) / (1024 ** 3)
    print(f"DONE: {OUT}  ({os.path.getsize(OUT):,} bytes = {gb:.2f} GB)  {total:,} docs")


if __name__ == "__main__":
    main()
