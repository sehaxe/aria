"""Aria-JEPA anti-collapse verification.

Guards against latent collapse (the standard JEPA failure): the predictor must
not degenerate to a constant, and VICReg (variance/covariance) must actively
penalize collapse. Also checks context-mask correctness and end-to-end finiteness.
"""
import sys
import torch

sys.path.insert(0, 'src')
from model.jepa import AriaJEPA, variance_loss, covariance_loss
from model.model import AriaModel


def test_build_context_mask_no_image():
    jepa = AriaJEPA(d_model=16, patch_size=8, context_keep=0.7)
    B, T = 2, 64
    patches = torch.randint(0, 255, (B, T, 16)).float()
    is_img = torch.zeros(B, T, dtype=torch.bool)
    tm, keep, C, T2 = jepa.build_context_mask(patches, is_img)
    assert tm.shape == (B, T) and T2 == T
    assert keep.shape == (B, T)
    assert tm.sum() > 0 and tm.sum() < B * T
    # image positions are never masked
    is_img2 = torch.zeros(B, T, dtype=torch.bool)
    is_img2[:, 8:16] = True
    tm2, _, _, _ = jepa.build_context_mask(patches, is_img2)
    assert tm2[:, 8:16].sum() == 0


def test_jepa_loss_finite_and_vicreg():
    # CPU-safe: call jepa_loss directly (no KAN / Triton kernel in the path).
    jepa = AriaJEPA(d_model=16)
    pk = torch.randn(1, 12, 16)
    pl = torch.randn(1, 12, 16)
    rep = torch.randn(1, 12, 16)
    tg = torch.randn(1, 12, 16)
    k, l, v, c = jepa.jepa_loss(pk, pl, rep, tg)
    for t in (k, l, v, c):
        assert torch.isfinite(t)
    assert v >= 0  # variance term is non-negative (fires only on collapse)


def test_variance_collapse_penalty():
    # constant input -> std 0 -> variance_loss = full eps penalty
    x = torch.zeros(8, 16)
    assert variance_loss(x) > 0
    # spread-out input -> variance penalty near zero
    x2 = torch.randn(2048, 16)
    assert variance_loss(x2) < variance_loss(x)


def test_covariance_relative():
    # collapsed (redundant) features -> high off-diagonal covariance -> high loss.
    # well-spread random reps have near-zero off-diagonal covariance.
    x = torch.randn(2048, 16)
    collapsed = x[:, :1].expand(-1, 16)  # all 16 dims identical -> heavy redundancy
    assert covariance_loss(collapsed) > covariance_loss(x)


def test_jepa_only_no_collapse():
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
    for _ in range(3):
        loss = m(patches, lengths, is_img)
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        opt.zero_grad()
    with torch.no_grad():
        tm, _, _, _ = m.jepa.build_context_mask(patches, is_img, lengths)
        _, _, _, h_on, _ = m.run_encoded(m.encoder(patches, is_img), patches, is_img,
                                active_loops=3, patch_mask=tm, return_hidden=True)
        on = h_on[tm]
        pk, pl, gate, rep = m.jepa(on.unsqueeze(0))
        spread = (pk.squeeze(0).max(0).values - pk.squeeze(0).min(0).values).mean()
        assert spread > 0, "predictor collapsed to a constant"
        # router must engage both predictors (not saturated to all-0 / all-1)
        assert 0.05 < gate.mean().item() < 0.95


if __name__ == "__main__":
    test_build_context_mask_no_image()
    test_jepa_loss_finite_and_vicreg()
    test_variance_collapse_penalty()
    test_covariance_relative()
    test_jepa_only_no_collapse()
    print("JEPA collapse tests: SUCCESS")
