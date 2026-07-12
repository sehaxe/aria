import torch
import pytest
from kernels.sct_quant import FusedSCTQuant


def ref_forward(U, ternary):
    # au is treated as a constant (matches sct.SCTLinear's U / au.detach())
    au = (U.abs().mean() + 1e-12).detach()
    uq = torch.clamp(U / au, -1.0, 1.0)
    if ternary:
        ut = torch.where(uq > 0.5, torch.ones_like(uq),
                         torch.where(uq < -0.5, -torch.ones_like(uq), torch.zeros_like(uq)))
        uq = uq + (ut - uq).detach()
    return uq


def ref_backward(U, grad):
    au = U.abs().mean() + 1e-12
    us = U / au
    inside = (us > -1.0) & (us < 1.0)
    return torch.where(inside, grad / au, torch.zeros_like(grad))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel needs CUDA")
def test_sct_quant_forward_matches_reference():
    torch.manual_seed(0)
    U = torch.randn(64, 32, device="cuda", dtype=torch.float32)
    for ternary in (True, False):
        out = FusedSCTQuant.apply(U, ternary)
        ref = ref_forward(U, ternary)
        assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max().item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel needs CUDA")
def test_sct_quant_backward_matches_reference():
    torch.manual_seed(0)
    U = (torch.randn(64, 32, device="cuda") * 2.0).requires_grad_(True)
    U2 = U.detach().clone().requires_grad_(True)
    FusedSCTQuant.apply(U, True).sum().backward()
    ref_forward(U2, True).sum().backward()
    assert torch.allclose(U.grad, U2.grad, atol=1e-5), (U.grad - U2.grad).abs().max().item()
