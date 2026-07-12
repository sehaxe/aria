import sys, torch
sys.path.insert(0, 'src')
from model.model import AriaModel
from model.helix import HelixCore
from model.hwf_kan import DynamicFFN

torch.manual_seed(42)
device = 'cuda'

# Test HelixCore in isolation
helix = HelixCore(64, max_loops=6, degree=6, num_frequencies=3).cuda()
x = torch.randn(2, 8, 64, device='cuda')
acc, halt = helix(x)
print(f"HelixCore input norm: {x.norm().item():.4f}")
print(f"HelixCore output norm: {acc.norm().item():.4f}")
print(f"Halt steps: {len(halt)}")
for i, hp in enumerate(halt):
    print(f"  step {i}: halt_prob mean={hp.mean().item():.4f}, min={hp.min().item():.4f}, max={hp.max().item():.4f}")

# Check if output depends on input
x2 = torch.randn_like(x)
acc2, _ = helix(x2)
print(f"Different input -> different output: {(acc - acc2).norm().item():.4f}")

# Check gradients
loss = acc.mean()
loss.backward()
for n, p in helix.named_parameters():
    if p.grad is not None and p.grad.abs().sum().item() > 0:
        print(f"  {n}: grad ok")
        break
