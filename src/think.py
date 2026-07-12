"""Recurrent "thinking" generation, gradio-free. Successor to the old ui.py; used by the Textual TUI."""
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import yaml

PAD_ID = 268
MAX_PATCH_LEN = 16


def encode_prompt(text, device, max_patch_len=MAX_PATCH_LEN, pad_id=PAD_ID):
    data = text.encode("utf-8")
    chunks = [data[i:i + max_patch_len] for i in range(0, max(len(data), 1), max_patch_len)]
    T = len(chunks)
    patches = torch.zeros(1, T, 768, device=device)
    for t, ch in enumerate(chunks):
        if ch:
            patches[0, t, 0:len(ch)] = torch.tensor(list(ch), dtype=torch.float32, device=device)
    is_image = torch.zeros(1, T, dtype=torch.bool, device=device)
    lengths = torch.ones(1, T, dtype=torch.float32, device=device)
    return patches, is_image, lengths


def compute_depth(halt_probs, pos, max_loops):
    """Effective recurrent depth at patch pos: sum(step * p_step * remaining), a weighted halt-prob sum."""
    depth = 0.0
    remaining = 1.0
    for step_idx, p_step in enumerate(halt_probs):
        p_val = float(p_step[0, pos, 0].item())
        depth += (step_idx + 1) * p_val * remaining
        remaining *= (1.0 - p_val)
    return depth


@torch.no_grad()
def generate(model, text, max_new_bytes=128, temp=0.7, max_patch_len=MAX_PATCH_LEN,
             pad_id=PAD_ID, max_loops=6, device="cuda"):
    model.eval()
    patches, is_image, lengths = encode_prompt(text, device, max_patch_len, pad_id)
    pairs = [(b, 1.0) for b in text.encode("utf-8")]
    max_patches = max(1, (max_new_bytes + max_patch_len - 1) // max_patch_len)
    for _ in range(max_patches):
        logits, _, halt_probs = model(patches, lengths, is_image, return_halt=True)
        if logits is None:
            break
        last = logits[:, -1]
        next_bytes = []
        for j in range(max_patch_len):
            probs = F.softmax(last[0, j] / temp, dim=-1)
            next_bytes.append(int(torch.multinomial(probs, 1).item()))
        pos = patches.shape[1] - 1
        depth = compute_depth(halt_probs, pos, max_loops)
        for b in next_bytes:
            pairs.append((b, depth))
        np_ = torch.zeros(1, 1, 768, device=device)
        np_[0, 0, 0:max_patch_len] = torch.tensor(next_bytes, dtype=torch.float32, device=device)
        patches = torch.cat([patches, np_], dim=1)
        is_image = torch.cat([is_image, torch.zeros(1, 1, dtype=torch.bool, device=device)], dim=1)
        lengths = torch.cat([lengths, torch.ones(1, 1, dtype=torch.float32, device=device)], dim=1)
        if 10 in next_bytes:
            break
    return pairs


_SRC = Path(__file__).resolve().parent

def build_think_model(config_path=None, checkpoint_path=None, device=None):
    config_path = config_path or str(_SRC.parent / "configs" / "29m.yaml")
    checkpoint_path = checkpoint_path or str(_SRC.parent / "checkpoint.pt")
    from model.model import AriaModel
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AriaModel(
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_loops=cfg["n_loops"],
        rank=cfg.get("sct_rank", 32), nsa=cfg.get("nsa", False),
        window_size=cfg.get("window_size", 512), degree=cfg.get("degree", 6),
        num_frequencies=cfg.get("num_frequencies", 3), temperature=cfg.get("temperature", 1.0),
        ponder_lambda=cfg.get("ponder_lambda", 0.01), max_sigma=cfg.get("max_sigma", 1.0),
        compile=False,
    ).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"[Aria] WARNING: checkpoint {checkpoint_path} not found; random init.")
    model.eval()
    return model, cfg["n_loops"], device
