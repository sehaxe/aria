import torch, sys, time
sys.path.insert(0, 'src')
from model.model import AriaModel

def test_eager():
    m = AriaModel(d_model=256, n_heads=4, n_loops=4, vocab_size=256).cuda()
    B, N, L = 2, 4, 768
    patches = torch.randint(0, 255, (B, N, L)).float().cuda()
    lengths = torch.randint(1, 16, (B, N)).float().cuda()
    is_image_mask = torch.zeros(B, N, dtype=torch.bool).cuda()
    targets = torch.randint(0, 255, (B, N, 16)).long().cuda()
    loss = m(patches, lengths, is_image_mask, targets=targets)
    loss.backward()
    n = sum(p.numel() for p in m.parameters())
    print(f"EAGER Params: {n:,} | Loss: {loss.item():.4f}")
    assert 0.5 < loss.item() < 10

if __name__ == '__main__':
    test_eager()
    print("EAGER PASS")
