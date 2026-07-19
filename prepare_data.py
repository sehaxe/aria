"""Pack text files into a single .bin (uint8) for mmap streaming.

Usage:
    python prepare_data.py data/pretrain/engram/ data/pretrain/engram/wikipedia.bin
    python prepare_data.py data/pretrain/reasoning/ data/pretrain/reasoning.bin
"""
import os, sys, random
from pathlib import Path


def pack_text_folder_to_bin(input_dir, output_bin_path):
    files = (
        list(Path(input_dir).rglob("*.txt"))
        + list(Path(input_dir).rglob("*.json"))
        + list(Path(input_dir).rglob("*.md"))
    )
    files = [f for f in files if f.stat().st_size > 0]
    if not files:
        print(f"No text files found in {input_dir}")
        return

    # ponytail: shuffle file order (each file = one ~6MB chunk), never the
    # contents, so RAM stays at O(one chunk) instead of O(corpus). Stream write.
    random.shuffle(files)

    os.makedirs(Path(output_bin_path).parent, exist_ok=True)
    sep = b"\n"
    size = 0
    with open(output_bin_path, "wb") as out:
        for f in files:
            try:
                data = f.read_bytes()
            except Exception as e:
                print(f"  skip {f.name}: {e}")
                continue
            if not data:
                continue
            if data[-1:] != sep:
                data += sep
            out.write(data)
            size += len(data)

    gb = size / (1024 ** 3)
    print(f"Written: {output_bin_path}  ({size:,} bytes = {gb:.2f} GB)  from {len(files):,} files")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python prepare_data.py <input_dir> <output_bin_path>")
        sys.exit(1)
    pack_text_folder_to_bin(sys.argv[1], sys.argv[2])
