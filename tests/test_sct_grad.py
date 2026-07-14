"""SCT backward gradient correctness (validates the `*s` fix in sct.py).

The analytic backward stored `dx = (g @ v) @ u.T`, missing the `*s` factor
from `hs = h * s`. This silently corrupts gradient flow during training.
These tests compare the custom backward to a plain-torch reference and assert
the correct `dx` actually contains the `*s` scale (not the buggy version).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch

from model.sct import _SCTMM, _BitnetSCTMM, _Fp8SCTMM
from bitnet_v2 import _pack_int4, _dequant, _HadamardTransform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="SCT Triton kernel requires CUDA"
)

DEVICE = "cuda"
DTYPE = torch.float32


def _ref_plain(x, u, s, v, gamma):
    """Plain torch reimplementation of z = ((x@u)*s)@v.T * gamma."""
    z = ((x @ u) * s) @ v.T * gamma
    z.sum().backward()
    return [x.grad, u.grad, s.grad, v.grad, gamma.grad]


def test_sct_mm_backward_matches_reference():
    M, K, R, N = 6, 10, 5, 8
    x = torch.randn(M, K, device=DEVICE, dtype=DTYPE, requires_grad=True)
    u = torch.randn(K, R, device=DEVICE, dtype=DTYPE, requires_grad=True)
    s = torch.randn(R, device=DEVICE, dtype=DTYPE, requires_grad=True)
    v = torch.randn(N, R, device=DEVICE, dtype=DTYPE, requires_grad=True)
    gamma = torch.tensor(1.4, device=DEVICE, dtype=DTYPE, requires_grad=True)

    out = _SCTMM.apply(x, u, s, v, gamma)
    out.sum().backward()
    gx, gu, gs, gv, gg = x.grad, u.grad, s.grad, v.grad, gamma.grad

    rx, ru, rs, rv, rg = _ref_plain(
        x.detach().clone().requires_grad_(True),
        u.detach().clone().requires_grad_(True),
        s.detach().clone().requires_grad_(True),
        v.detach().clone().requires_grad_(True),
        gamma.detach().clone().requires_grad_(True),
    )
    assert torch.allclose(gx, rx, atol=1e-4), (gx - rx).abs().max()
    assert torch.allclose(gu, ru, atol=1e-4)
    assert torch.allclose(gs, rs, atol=1e-4)
    assert torch.allclose(gv, rv, atol=1e-4)
    assert torch.allclose(gg, rg, atol=1e-4)

    # invariant: correct dx MUST include the *s factor (Bug1 fix)
    g = torch.ones_like(out) * gamma.detach()
    buggy_dx = (g @ v.detach()) @ u.detach().T  # missing *s
    assert not torch.allclose(gx, buggy_dx, atol=1e-6)


def test_bitnet_sct_mm_backward_matches_reference():
    M, K, R, N = 6, 10, 5, 8
    bits, had = 4, True
    x = torch.randn(M, K, device=DEVICE, dtype=DTYPE, requires_grad=True)
    u = torch.randn(K, R, device=DEVICE, dtype=DTYPE, requires_grad=True)
    s = torch.randn(R, device=DEVICE, dtype=DTYPE, requires_grad=True)
    v = torch.randn(N, R, device=DEVICE, dtype=DTYPE, requires_grad=True)
    gamma = torch.tensor(1.4, device=DEVICE, dtype=DTYPE, requires_grad=True)

    out = _BitnetSCTMM.apply(x, u, s, v, gamma, bits, had)
    out.sum().backward()
    gx = x.grad

    # STE reference mirroring the forward's inline quant (Hadamard applied ONCE,
    # then quantized). grad to x flows via STE through the quant + the Hadamard.
    xr = x.detach().clone().requires_grad_(True)
    xh = _HadamardTransform.apply(xr)
    if bits >= 8:
        scale = xh.abs().amax(-1, keepdim=True).clamp(min=1e-12)
        q = (xh / scale * 127).round().clamp(-128, 127).to(torch.int8)
        packed = False
    else:
        scale = xh.abs().mean(-1, keepdim=True).clamp(min=1e-12)
        q = (xh / scale * 7).round().clamp(-8, 7).to(torch.int8)
        packed = xh.shape[-1] % 2 == 0
        if packed:
            q = _pack_int4(q)
    deq = _dequant(q, scale, bits, packed)
    deq_ste = xh + (deq - xh).detach()  # STE: grad to xh = identity
    z = ((deq_ste @ u.detach()) * s.detach()) @ v.detach().T * gamma.detach()
    z.sum().backward()
    gx_ref = xr.grad  # = H @ dz/d(deq), matches the backward's trailing Hadamard
    assert torch.allclose(gx, gx_ref, atol=1e-3), (gx - gx_ref).abs().max()

    # invariant: correct dx MUST include the *s factor (Bug1 fix)
    g = torch.ones_like(out) * gamma.detach()
    buggy_dx = (g @ v.detach()) @ u.detach().T
    assert not torch.allclose(gx, buggy_dx, atol=1e-6)


def test_fp8_sct_mm_backward_finite_and_scaled():
    M, K, R, N = 6, 16, 16, 16  # all matmul dims must be multiples of 16 for fp8 scaled_mm
    x = torch.randn(M, K, device=DEVICE, dtype=DTYPE, requires_grad=True)
    u = torch.randn(K, R, device=DEVICE, dtype=DTYPE, requires_grad=True)
    s = torch.randn(R, device=DEVICE, dtype=DTYPE, requires_grad=True)
    v = torch.randn(N, R, device=DEVICE, dtype=DTYPE, requires_grad=True)
    gamma = torch.tensor(1.4, device=DEVICE, dtype=DTYPE, requires_grad=True)

    out = _Fp8SCTMM.apply(x, u, s, v, gamma)
    out.sum().backward()
    gx = x.grad
    assert torch.isfinite(gx).all()

    # fp8 scales cancel (a = x/sa etc.), so dx ~= equivalent _SCTMM dx
    xr = x.detach().clone().requires_grad_(True)
    outr = _SCTMM.apply(xr, u.detach(), s.detach(), v.detach(), gamma.detach())
    outr.sum().backward()
    assert torch.allclose(gx, xr.grad, atol=5e-2), (gx - xr.grad).abs().max()

    # invariant: correct dx MUST include the *s factor (Bug1 fix)
    g = torch.ones_like(out) * gamma.detach()
    buggy_dx = (g @ v.detach()) @ u.detach().T
    assert not torch.allclose(gx, buggy_dx, atol=1e-6)


if __name__ == "__main__":
    test_sct_mm_backward_matches_reference()
    test_bitnet_sct_mm_backward_matches_reference()
    test_fp8_sct_mm_backward_finite_and_scaled()
    print("SCT grad tests: SUCCESS")
