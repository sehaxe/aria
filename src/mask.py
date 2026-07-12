import torch

def combined_mask(n_loops, dropbp_prob, backward_ratio, device="cpu"):
    # ponytail: must build on-device; a CPU mask can't be mixed into CUDA-graph
    # capture (CPU<->CUDA copies are illegal there). helix passes x.device.
    mask = torch.zeros(n_loops, dtype=torch.bool, device=device)
    if dropbp_prob > 0:
        mask = mask | (torch.rand(n_loops, device=device) < dropbp_prob)
    if backward_ratio < 1:
        keep_n = max(1, int(n_loops * backward_ratio))
        idx = torch.randperm(n_loops, device=device)[:keep_n]
        skip = torch.ones(n_loops, dtype=torch.bool, device=device)
        # torch.tensor(False, device=device) copies CPU->CUDA (illegal in graph
        # capture); use a zero tensor (no scalar payload) + scatter instead.
        skip = skip.scatter(0, idx, torch.zeros(keep_n, dtype=torch.bool, device=device))
        mask = mask | skip
    # last loop is never skipped (full BPTT to the final state) — re-append a
    # device zero without a scalar assignment.
    mask = torch.cat([mask[:-1], torch.zeros(1, dtype=torch.bool, device=device)])
    return mask
