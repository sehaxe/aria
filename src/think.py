"""Recurrent "thinking" generation, gradio-free. Successor to the old ui.py; used by the Textual TUI."""
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import yaml
from spectral_tta import spectral_tta_step, restore_spectral_tta

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


def _patch_tensor(bytes_list, device):
    t = torch.zeros(1, 1, 768, device=device)
    t[0, 0, 0:len(bytes_list)] = torch.tensor(bytes_list, dtype=torch.float32, device=device)
    return t


def _sample_patch(logits, temp):
    """Sample a 16-byte patch from per-position logits [16, vocab].

    Only the first 256 classes are valid bytes; special tokens (256-268) are
    padding/internal and must never be emitted during generation.
    """
    out = []
    for j in range(logits.shape[0]):
        p = F.softmax(logits[j, :256] / temp, dim=-1)
        out.append(int(torch.multinomial(p, 1).item()))
    return out


@torch.no_grad()
def _speculative_step(model, h_last, patches, is_image, lengths, K, temp, device):
    """Self-speculative decode one draft round.

    Draft K patches with the recurrent core's draft head (cheap), then verify
    them in ONE full forward and accept via residual (rejection) sampling.
    Returns list of (bytes[16], depth) accepted, or [] if verification failed.
    """
    h_prev = h_last                                   # (1,1,D) trunk hidden at last ctx pos
    ctx_emb = model.encoder(patches[:, -1:], is_image[:, -1:])  # (1,1,D) last real patch
    draft_patches, draft_logits = [], []
    x_latent = None
    for k in range(K):
        h_in = h_prev + (ctx_emb if k == 0 else x_latent)
        h_next = model.draft_hidden(h_in)
        lq = model.decoder(h_next, torch.zeros(1, 1, dtype=torch.bool, device=device))[0][0, -1]  # (16, vocab)
        draft_logits.append(lq)
        x = _sample_patch(lq, temp)
        draft_patches.append(x)
        xb = _patch_tensor(x, device)
        x_latent = model.encoder(xb, torch.zeros(1, 1, dtype=torch.bool, device=device))
        h_prev = h_next
    if not draft_patches:
        return []
    np_ = torch.cat([_patch_tensor(p, device) for p in draft_patches], dim=1)  # (1,K,768)
    full_patches = torch.cat([patches, np_], dim=1)
    full_isimg = torch.cat([is_image, torch.zeros(1, np_.shape[1], dtype=torch.bool, device=device)], dim=1)
    full_len = torch.cat([lengths, torch.ones(1, np_.shape[1], dtype=torch.float32, device=device)], dim=1)
    vlogits, _, vhalt, _, _ = model(full_patches, full_len, full_isimg, return_halt=True, return_hidden=True)
    T = patches.shape[1]
    # p[k] = true dist at position T-1+k (predicts drafted patch k); q = draft dist.
    p = F.softmax(vlogits[0, T - 1:T + K - 1] / temp, dim=-1)   # (K,16,vocab)
    q = F.softmax(torch.stack(draft_logits, 0) / temp, dim=-1)  # (K,16,vocab)
    accepted = []
    for k in range(K):
        ok = True
        for j in range(p.shape[1]):
            x = draft_patches[k][j]
            u = torch.rand(1).item()
            if u >= min(1.0, p[k, j, x].item() / max(q[k, j, x].item(), 1e-12)):
                ok = False
                break
        if ok:
            accepted.append(draft_patches[k])
            continue
        # rejected: resample the whole patch from the corrected dist, append, stop.
        corr = (p[k] - q[k]).clamp_min(0)              # (16,vocab)
        pc = corr / corr.sum(-1, keepdim=True).clamp_min(1e-12)
        accepted.append([int(torch.multinomial(pc[j], 1).item()) for j in range(16)])
        break
    if len(accepted) == K:
        accepted.append(_sample_patch(vlogits[0, T + K - 1], temp))  # bonus true token
    vpos = min(T + len(accepted) - 1, vhalt[0].shape[0] - 1) if vhalt else 0
    depth = compute_depth(vhalt, vpos, model.speculative_k + 1) if vhalt else 1.0
    return [(p_, depth) for p_ in accepted]


@torch.no_grad()
def generate(model, text, max_new_bytes=128, temp=0.7, max_patch_len=MAX_PATCH_LEN,
             pad_id=PAD_ID, max_loops=6, device="cuda", speculative=None, tta_steps=0):
    model.eval()
    patches, is_image, lengths = encode_prompt(text, device, max_patch_len, pad_id)
    pairs = [(b, 1.0) for b in text.encode("utf-8")]
    max_patches = max(1, (max_new_bytes + max_patch_len - 1) // max_patch_len)
    use_spec = bool(model.speculative) if speculative is None else bool(speculative)
    K = model.speculative_k if use_spec else 0
    try:
        if tta_steps and tta_steps > 0:
            # ponytail: adapt singular scales to the prompt, persist into the
            # generation below; finally reverts so the model stays clean for
            # the next turn.
            spectral_tta_step(model, patches, lengths, is_image, steps=tta_steps, persist=True)
        for _ in range(max_patches):
            out = model(patches, lengths, is_image, return_halt=True, return_hidden=True)
            logits, pixels, halt_probs, h, _ = out
            if logits is None:
                break
            last = logits[:, -1]
            pos = patches.shape[1] - 1
            depth = compute_depth(halt_probs, pos, max_loops)
            if K and K > 0 and h is not None:
                accepted = _speculative_step(model, h[:, -1:], patches, is_image, lengths,
                                             K, temp, device)
                if not accepted:
                    break
                for pb, pd in accepted:
                    pairs.extend((b, pd) for b in pb)
                    np_ = _patch_tensor(pb, device)
                    patches = torch.cat([patches, np_], dim=1)
                    is_image = torch.cat([is_image, torch.zeros(1, 1, dtype=torch.bool, device=device)], dim=1)
                    lengths = torch.cat([lengths, torch.ones(1, 1, dtype=torch.float32, device=device)], dim=1)
                    if 10 in pb:
                        return pairs
                continue
            # fallback: one patch per forward (no speculation / draft head absent)
            next_bytes = []
            for j in range(max_patch_len):
                probs = F.softmax(last[0, j, :256] / temp, dim=-1)
                next_bytes.append(int(torch.multinomial(probs, 1).item()))
            for b in next_bytes:
                pairs.append((b, depth))
            np_ = torch.zeros(1, 1, 768, device=device)
            np_[0, 0, 0:max_patch_len] = torch.tensor(next_bytes, dtype=torch.float32, device=device)
            patches = torch.cat([patches, np_], dim=1)
            is_image = torch.cat([is_image, torch.zeros(1, 1, dtype=torch.bool, device=device)], dim=1)
            lengths = torch.cat([lengths, torch.ones(1, 1, dtype=torch.float32, device=device)], dim=1)
            if 10 in next_bytes:
                break
    finally:
        if tta_steps and tta_steps > 0:
            restore_spectral_tta(model)
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
        rank=cfg.get("sct_rank", 32),         nsa=cfg.get("nsa", False), nsa_every=cfg.get("nsa_every", 3),
        window_size=cfg.get("window_size", 512), degree=cfg.get("degree", 6),
        num_frequencies=cfg.get("num_frequencies", 3), temperature=cfg.get("temperature", 1.0),
        ponder_lambda=cfg.get("ponder_lambda", 0.01), max_sigma=cfg.get("max_sigma", 1.0),
        adaptive_loops=cfg.get("adaptive_loops", False),
        speculative=cfg.get("speculative", True), speculative_k=cfg.get("speculative_k", 4),
        mtp=cfg.get("mtp", True), mtp_k=cfg.get("mtp_k", 4), mtp_loss_coef=cfg.get("mtp_loss_coef", 0.1),
        worldmodel_halt=cfg.get("worldmodel_halt", True),
        compile=False,
    ).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"[Aria] WARNING: checkpoint {checkpoint_path} not found; random init.")
    model.eval()
    return model, cfg["n_loops"], device
