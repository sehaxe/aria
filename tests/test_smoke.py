"""Smoke test: byte-level forward + backward, inference path."""
import sys
import torch

sys.path.insert(0, 'src')
from model.model import AriaModel


def test_byte_level_smoke():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AriaModel(
        d_model=128,
        n_heads=2,
        n_loops=2,
        rank=8,
        nsa=False,
    ).to(device)

    # Batch of text patches: 16 bytes each, padded to 768-dim encoder input.
    B, N, L = 2, 4, 768
    patches = torch.randint(0, 255, (B, N, L)).float().to(device)
    lengths = torch.randint(1, 16, (B, N)).float().to(device)
    is_image_mask = torch.zeros(B, N, dtype=torch.bool).to(device)  # 2D (B, N)
    targets = torch.randint(0, 255, (B, N, 16)).long().to(device)

    model.train()
    loss = model(patches, lengths, is_image_mask, targets=targets)
    assert loss is not None
    assert torch.isfinite(loss)
    loss.backward()

    model.eval()
    logits, pixels = model(patches, lengths, is_image_mask)
    assert logits is not None or pixels is not None
    print("Smoke test: SUCCESS")


if __name__ == "__main__":
    test_byte_level_smoke()
