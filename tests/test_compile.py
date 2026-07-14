import torch, sys, time
sys.path.insert(0, 'src')
from model.model import AriaModel

def test_eager():
    m = AriaModel(d_model=256, n_heads=4, n_loops=4, vocab_size=256).cuda()
    x = torch.randint(0, 256, (2, 128)).cuda()
    y = torch.randint(0, 256, (2, 131)).cuda()
    loss = m(x, targets=y)
    loss.backward()
    n = sum(p.numel() for p in m.parameters())
    print(f"EAGER Params: {n:,} | Loss: {loss.item():.4f}")
    assert 0.5 < loss.item() < 10

if __name__ == '__main__':
    test_eager()
    print("EAGER PASS")
