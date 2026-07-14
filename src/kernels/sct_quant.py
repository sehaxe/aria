import torch
import triton
import triton.language as tl


@triton.jit
def _sct_quant_fwd_kernel(
    u_ptr, uq_ptr,
    au_ptr,             # device pointer to mean-abs scale (1-elem tensor)
    numel,
    ternary: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    au = tl.load(au_ptr)
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    u = tl.load(u_ptr + offsets, mask=mask).to(tl.float32)
    u_scaled = u / au
    u_clamped = tl.minimum(tl.maximum(u_scaled, -1.0), 1.0)
    if ternary:
        u_out = tl.where(u_clamped > 0.5, 1.0,
                         tl.where(u_clamped < -0.5, -1.0, 0.0))
    else:
        u_out = u_clamped
    tl.store(uq_ptr + offsets, u_out.to(uq_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _sct_quant_bwd_kernel(
    u_ptr, g_ptr, gq_ptr,
    au_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    au = tl.load(au_ptr)
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    u = tl.load(u_ptr + offsets, mask=mask).to(tl.float32)
    g = tl.load(g_ptr + offsets, mask=mask).to(tl.float32)
    # STE through the clamp: grad passes as 1/au inside [-1,1], zeroed outside.
    us = u / au
    inside = (us > -1.0) & (us < 1.0)
    gq = tl.where(inside, g / au, 0.0)
    tl.store(gq_ptr + offsets, gq.to(gq_ptr.dtype.element_ty), mask=mask)


class FusedSCTQuant(torch.autograd.Function):
    """Monolithic ternary quantization for SCTLinear factors.

    Forward matches sct.SCTLinear's reference math (mean-abs normalize, clamp
    to [-1,1], optional ternary hardening). Backward is the straight-through
    estimator of that clamp. Runs in one pass, no intermediate tensor allocs.
    au is kept as a device tensor (loaded inside the kernel) so CUDA graph
    capture is not broken by a .item() CPU sync.
    """

    @staticmethod
    def forward(ctx, U, ternary=True):
        shape = U.shape
        flat_U = U.reshape(-1).contiguous()
        numel = flat_U.numel()
        au = (U.abs().mean() + 1e-12).reshape(1)  # device tensor, graph-safe
        out = torch.empty_like(flat_U)
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(numel, BLOCK_SIZE),)
        _sct_quant_fwd_kernel[grid](flat_U, out, au, numel, ternary, BLOCK_SIZE)
        ctx.save_for_backward(U)
        ctx.au = au
        return out.view(*shape)

    @staticmethod
    def backward(ctx, grad_out):
        U = ctx.saved_tensors[0]
        au = ctx.au
        flat_g = grad_out.reshape(-1).contiguous()
        numel = flat_g.numel()
        gout = torch.empty_like(flat_g)
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(numel, BLOCK_SIZE),)
        _sct_quant_bwd_kernel[grid](U.reshape(-1).contiguous(), flat_g, gout, au, numel, BLOCK_SIZE)
        return gout.view_as(U), None
