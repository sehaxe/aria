import torch
import torch.nn.functional as F


@torch.no_grad()
def compress_prompt(model, input_ids, target_info_ratio=0.5):
    """Entropy-based prompt compressor for the pure-byte Aria architecture.

    Entropy is measured from the autoregressive byte-decoder's output logits
    over the 269-byte vocab; highest-entropy patches are trimmed first to hit
    the target information budget.
    """
    B, T = input_ids.shape
    device = input_ids.device

    # Pack flat bytes into 16-byte dynamic patches
    max_len = 16
    pad_len = (max_len - (T % max_len)) % max_len
    padded_ids = F.pad(input_ids, (0, pad_len), value=268) if pad_len > 0 else input_ids

    N = padded_ids.shape[1] // max_len
    patches = padded_ids.view(B, N, max_len)

    # Pad to the unified 768-dim encoder input; pad token is 268
    patches_768 = F.pad(patches, (0, 768 - max_len), value=268).float().to(device)
    patch_lengths = (patches != 268).sum(dim=-1).float().to(device)
    is_image_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

    # Forward pass -> byte logits (B, N, 16, 269)
    logits, _ = model(patches_768, patch_lengths, is_image_mask)

    probs = F.softmax(logits, dim=-1)
    # Mean entropy within a patch, then across the batch -> (N,)
    entropy = -(probs * probs.log().clamp_min(1e-12)).sum(dim=-1).mean(dim=-1).mean(dim=0)

    total_h = entropy.sum()
    target_h = total_h * target_info_ratio
    cum = 0.0
    cut = N
    for i in range(N - 1, 0, -1):
        cum += entropy[i].item()
        if cum >= target_h.item():
            cut = i
            break

    # Return the trimmed patches unrolled back to a flat sequence
    cut_ids = patches[:, :cut].flatten().unsqueeze(0)
    return cut_ids, cut / N
