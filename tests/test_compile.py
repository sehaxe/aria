import torch, sys, time
sys.path.insert(0, 'src')
from model.model import AriaModel

def test_eager():
    m = AriaModel(d_model=256, n_heads=4, n_loops=4, vocab_size=256, use_cld=False).cuda()
    x = torch.randint(0, 256, (2, 128)).cuda()
    y = torch.randint(0, 256, (2, 131)).cuda()
    loss = m(x, targets=y)
    loss.backward()
    n = sum(p.numel() for p in m.parameters())
    print(f"EAGER Params: {n:,} | Loss: {loss.item():.4f}")
    assert 0.5 < loss.item() < 10

def test_compiled_anchor():
    """Compile only the anchor block (FlexAttention)."""
    from model.anchor import AnchorBlock
    m = AnchorBlock(256, 4).cuda()
    m = torch.compile(m)
    x = torch.randn(2, 128, 256, device='cuda')
    t0 = time.time()
    out = m(x)
    print(f"COMPILED ANCHOR: {out.shape} in {time.time()-t0:.1f}s")

if __name__ == '__main__':
    test_eager()
    print("EAGER PASS")
    test_compiled_anchor()
    print("ALL PASS")
