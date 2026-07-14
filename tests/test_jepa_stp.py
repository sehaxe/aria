"""Aria-JEPA Semantic Tube Prediction (STP) verification.

STP is a curvature regularizer on the HelixCore loop trajectory: it penalizes
the 2nd-difference (acceleration) across loop steps, so a perfectly linear
trajectory costs 0 and a curving one costs >0. Verifies the math + integration.
"""
import sys
import torch

sys.path.insert(0, 'src')
from model.jepa import AriaJEPA


def _traj(coeffs, L, B=2, T=3, D=4):
    # coeffs: list of [D] tensors for a + t*b + t^2*c + ...
    out = []
    for t in range(L):
        v = torch.zeros(D)
        for p, c in enumerate(coeffs):
            v = v + (t ** p) * c
        out.append(v.unsqueeze(0).unsqueeze(0).expand(B, T, D).clone())
    return out


def test_semantic_tube_linear_zero():
    jepa = AriaJEPA(d_model=4)
    a, b = torch.randn(4), torch.randn(4)
    traj = _traj([a, b], L=6)
    assert float(jepa.semantic_tube(traj)) < 1e-5


def test_semantic_tube_quadratic_positive():
    jepa = AriaJEPA(d_model=4)
    c = torch.randn(4)
    traj = _traj([torch.zeros(4), torch.zeros(4), c], L=6)  # a + t^2*c
    assert float(jepa.semantic_tube(traj)) > 0


def test_semantic_tube_short_traj_zero():
    jepa = AriaJEPA(d_model=4)
    a = torch.randn(4)
    traj = _traj([a], L=2)
    assert float(jepa.semantic_tube(traj)) == 0.0


def test_full_model_stp_finite():
    from model.model import AriaModel
    m = AriaModel(d_model=32, n_heads=2, n_loops=3, rank=8, nsa=False,
                  jepa=True, jepa_stp=0.1).cuda()
    m.jepa_active = True
    m.jepa_only = True
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    B, N = 2, 32
    patches = torch.randint(0, 255, (B, N, 768)).float().cuda()
    lengths = torch.full((B, N), 16.0).cuda()
    is_img = torch.zeros(B, N, dtype=torch.bool).cuda()
    loss = m(patches, lengths, is_img)
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()
    aux = m.last_jepa_aux
    assert 'stp' in aux and torch.isfinite(torch.tensor(aux['stp']))


def test_full_model_no_stp_zero_term():
    from model.model import AriaModel
    m = AriaModel(d_model=32, n_heads=2, n_loops=3, rank=8, nsa=False,
                  jepa=True, jepa_stp=0.0).cuda()
    m.jepa_active = True
    m.jepa_only = True
    m.train()
    B, N = 2, 32
    patches = torch.randint(0, 255, (B, N, 768)).float().cuda()
    lengths = torch.full((B, N), 16.0).cuda()
    is_img = torch.zeros(B, N, dtype=torch.bool).cuda()
    loss = m(patches, lengths, is_img)
    assert torch.isfinite(loss)
    assert m.last_jepa_aux['stp'] == 0.0


if __name__ == "__main__":
    test_semantic_tube_linear_zero()
    test_semantic_tube_quadratic_positive()
    test_semantic_tube_short_traj_zero()
    test_full_model_stp_finite()
    test_full_model_no_stp_zero_term()
    print("JEPA STP tests: SUCCESS")
