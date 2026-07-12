"""Pack text files into a single .bin (uint8) for mmap streaming.

Usage:
    python prepare_data.py data/pretrain/engram/ data/pretrain/engram/wikipedia.bin
    python prepare_data.py data/pretrain/reasoning/ data/pretrain/reasoning.bin
"""
import os, sys, random
from pathlib import Path
import numpy as np


def pack_text_folder_to_bin(input_dir, output_bin_path):
    print(f"Scanning {input_dir} for .txt / .json / .md ...")
    files = (
        list(Path(input_dir).rglob("*.txt"))
        + list(Path(input_dir).rglob("*.json"))
        + list(Path(input_dir).rglob("*.md"))
    )
    if not files:
        print(f"No text files found in {input_dir}")
        return

    documents = []
    for f in files:
        try:
            text = f.read_text("utf-8", errors="replace").strip()
            if text:
                documents.append(text)
        except Exception as e:
            print(f"  skip {f.name}: {e}")

    print(f"Documents: {len(documents):,}, shuffling ...")
    random.shuffle(documents)

    os.makedirs(Path(output_bin_path).parent, exist_ok=True)
    sep = b"\n"
    size = 0
    with open(output_bin_path, "wb") as out:
        for doc in documents:
            data = doc.encode("utf-8")
            out.write(data)
            out.write(sep)
            size += len(data) + len(sep)

    gb = size / (1024**3)
    print(f"Written: {output_bin_path}  ({size:,} bytes = {gb:.2f} GB)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python prepare_data.py <input_dir> <output_bin_path>")
        sys.exit(1)
    pack_text_folder_to_bin(sys.argv[1], sys.argv[2])
