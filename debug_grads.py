import sys, torch
sys.path.insert(0, 'src')
from model.model import AriaModel
import torch.nn.functional as F

torch.manual_seed(42)
device = 'cuda'
model = AriaModel(d_model=256, n_heads=4, n_loops=6, vocab_size=256,
                  degree=6, num_frequencies=3, temperature=1.0, ponder_lambda=0.01).cuda()

x = torch.randint(0, 256, (2, 32), device='cuda')
y = torch.randint(0, 256, (2, 33), device='cuda')

loss = model(x, targets=y)
loss.backward()

with torch.no_grad():
    emb = torch.nn.functional.embedding(x, model.embed_w)
    anc = model.anchor(emb)
    h, hp = model.helix(anc)
    logits = model.synth(h)
    print(f"Anchor output std: {anc.std().item():.4f}")
    print(f"Helix output std: {h.std().item():.4f}")
    print(f"Synth[0] logit std: {logits[0].std().item():.4f}")
    probs = torch.softmax(logits[0], dim=-1)
    print(f"Max prob: {probs.max().item():.4f}, min: {probs.min().item():.4f}")
