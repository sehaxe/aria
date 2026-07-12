import torch

def loop_embedding(step, dim, device):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    # multiply Python int step directly into the device tensor (no CPU->GPU copy, graph-safe)
    sinusoid_inp = step * inv_freq
    emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
    return emb.view(1, 1, dim)
