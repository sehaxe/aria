import torch

def make_sparse_68_mask(d_out, device='cuda'):
    n_groups = (d_out + 7) // 8
    full = n_groups * 8
    mask = torch.ones(full, device=device)
    rng = torch.randint(0, 8, (n_groups, 2), device=device)
    for g in range(n_groups):
        mask[g * 8 + rng[g, 0]] = 0
        if rng[g, 1] != rng[g, 0]:
            mask[g * 8 + rng[g, 1]] = 0
    return mask[:d_out]
