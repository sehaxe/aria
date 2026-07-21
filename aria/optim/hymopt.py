import torch

def ns(A, steps=5):
    # ponytail: canonical Muon Newton-Schulz (KellerJordan). MUST run in bf16 — the iteration is
    # numerically unstable in fp32 and diverges. Orthogonalizes up to a uniform scale (X@X.T ~ c*I),
    # which is all Muon needs. Upgrade: exact polar factor via SVD.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = A.to(torch.bfloat16)
    transpose = X.size(0) > X.size(1)
    if transpose:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        G = X @ X.T
        X = a * X + (b * G + c * (G @ G)) @ X
    if transpose:
        X = X.T
    return X.to(torch.float32)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-3, wd=0.1, momentum=0.95, rank=32):
        defaults = dict(lr=lr, wd=wd, momentum=momentum, rank=rank)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            lr, wd, mom, base_rank = g['lr'], g['wd'], g['momentum'], g['rank']
            for p in g['params']:
                if p.grad is None:
                    continue
                s = self.state[p]
                grad = p.grad
                if base_rank > 0 and p.ndim == 2:
                    d1, d2 = p.shape
                    r = min(base_rank, d1, d2)
                    if 'buf_p' not in s:
                        gf = grad.float()
                        s['buf_p'] = gf @ torch.randn(d2, r, device=p.device, dtype=gf.dtype)
                        s['buf_q'] = gf.T @ torch.randn(d1, r, device=p.device, dtype=gf.dtype)
                    bp, bq = s['buf_p'], s['buf_q']
                    gf = grad.float()
                    old_bp = bp.clone()
                    bp.mul_(mom).add_(gf @ bq)
                    bq.mul_(mom).add_(gf.T @ old_bp)
                    O = bp @ bq.T
                else:
                    if 'buf' not in s:
                        s['buf'] = torch.zeros_like(p)
                    s['buf'].mul_(mom).add_(grad.to(s['buf'].dtype))
                    O = s['buf']
                d = max(p.shape) if p.ndim > 0 else 1
                if O.ndim == 2 and base_rank >= d // 2:
                    O = ns(O.float(), steps=3).to(p.dtype)
                else:
                    O.div_(O.norm(dim=0, keepdim=True).clamp(min=1e-8))
                p.mul_(1.0 - lr * wd).add_(O.to(p.dtype), alpha=-lr * 0.2 * (d ** 0.5))
