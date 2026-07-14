import torch
import triton
import triton.language as tl


@triton.jit
def _gru_lti_fwd_kernel(
    px_ptr, ph_ptr, h_ptr, x_ptr, a_ptr, b_ptr, out_ptr,
    N, D,
    BLOCK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    offs_d = pid_b * BLOCK + tl.arange(0, BLOCK)
    mask = offs_d < D
    base = pid_n * D
    base3 = pid_n * 3 * D

    xr = tl.load(px_ptr + base3 + 0 * D + offs_d, mask=mask, other=0.0).to(tl.float32)
    xz = tl.load(px_ptr + base3 + 1 * D + offs_d, mask=mask, other=0.0).to(tl.float32)
    xn = tl.load(px_ptr + base3 + 2 * D + offs_d, mask=mask, other=0.0).to(tl.float32)
    hr = tl.load(ph_ptr + base3 + 0 * D + offs_d, mask=mask, other=0.0).to(tl.float32)
    hz = tl.load(ph_ptr + base3 + 1 * D + offs_d, mask=mask, other=0.0).to(tl.float32)
    hn = tl.load(ph_ptr + base3 + 2 * D + offs_d, mask=mask, other=0.0).to(tl.float32)
    hv = tl.load(h_ptr + base + offs_d, mask=mask, other=0.0).to(tl.float32)
    xv = tl.load(x_ptr + base + offs_d, mask=mask, other=0.0).to(tl.float32)
    av = tl.load(a_ptr + offs_d, mask=mask, other=0.0).to(tl.float32)
    bv = tl.load(b_ptr + offs_d, mask=mask, other=0.0).to(tl.float32)

    r = 1.0 / (1.0 + tl.exp(-(xr + hr)))
    z = 1.0 / (1.0 + tl.exp(-(xz + hz)))
    # tl.tanh does not exist in Triton 3.7.1 -> numerically stable tanh (no exp overflow)
    zz = xn + r * hn
    absz = tl.abs(zz)
    e = tl.exp(-2.0 * absz)
    n = tl.where(zz >= 0.0, (1.0 - e) / (1.0 + e), (e - 1.0) / (1.0 + e))
    h_cand = (1.0 - z) * n + z * hv
    h_next = av * h_cand + bv * xv

    tl.store(out_ptr + base + offs_d, h_next.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _gru_lti_bwd_kernel(
    px_ptr, ph_ptr, h_ptr, x_ptr, a_ptr, b_ptr, g_ptr,
    gx_ptr, gh_ptr, ghv_ptr, gxv_ptr,
    N, D,
    BLOCK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    offs_d = pid_b * BLOCK + tl.arange(0, BLOCK)
    mask = offs_d < D
    base = pid_n * D
    base3 = pid_n * 3 * D

    xr = tl.load(px_ptr + base3 + 0 * D + offs_d, mask=mask, other=0.0).to(tl.float32)
    xz = tl.load(px_ptr + base3 + 1 * D + offs_d, mask=mask, other=0.0).to(tl.float32)
    xn = tl.load(px_ptr + base3 + 2 * D + offs_d, mask=mask, other=0.0).to(tl.float32)
    hr = tl.load(ph_ptr + base3 + 0 * D + offs_d, mask=mask, other=0.0).to(tl.float32)
    hz = tl.load(ph_ptr + base3 + 1 * D + offs_d, mask=mask, other=0.0).to(tl.float32)
    hn = tl.load(ph_ptr + base3 + 2 * D + offs_d, mask=mask, other=0.0).to(tl.float32)
    hv = tl.load(h_ptr + base + offs_d, mask=mask, other=0.0).to(tl.float32)
    xv = tl.load(x_ptr + base + offs_d, mask=mask, other=0.0).to(tl.float32)
    av = tl.load(a_ptr + offs_d, mask=mask, other=0.0).to(tl.float32)
    bv = tl.load(b_ptr + offs_d, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + base + offs_d, mask=mask, other=0.0).to(tl.float32)

    r = 1.0 / (1.0 + tl.exp(-(xr + hr)))
    z = 1.0 / (1.0 + tl.exp(-(xz + hz)))
    zz = xn + r * hn
    absz = tl.abs(zz)
    e = tl.exp(-2.0 * absz)
    n = tl.where(zz >= 0.0, (1.0 - e) / (1.0 + e), (e - 1.0) / (1.0 + e))
    h_cand = (1.0 - z) * n + z * hv

    gh_cand = g * av
    gxv = g * bv
    ghv = gh_cand * z
    gn = gh_cand * (1.0 - z)
    gz = gh_cand * (hv - n)
    dn = 1.0 - n * n
    gu = gn * dn
    gxu = gu
    gr = gu * hn
    ghn = gu * r
    ds = gz * z * (1.0 - z)
    gxz = ds
    ghz = ds
    dt = gr * r * (1.0 - r)
    gxr = dt
    ghr = dt

    tl.store(gx_ptr + base3 + 0 * D + offs_d, gxr.to(gx_ptr.dtype.element_ty), mask=mask)
    tl.store(gx_ptr + base3 + 1 * D + offs_d, gxz.to(gx_ptr.dtype.element_ty), mask=mask)
    tl.store(gx_ptr + base3 + 2 * D + offs_d, gxu.to(gx_ptr.dtype.element_ty), mask=mask)
    tl.store(gh_ptr + base3 + 0 * D + offs_d, ghr.to(gh_ptr.dtype.element_ty), mask=mask)
    tl.store(gh_ptr + base3 + 1 * D + offs_d, ghz.to(gh_ptr.dtype.element_ty), mask=mask)
    tl.store(gh_ptr + base3 + 2 * D + offs_d, ghn.to(gh_ptr.dtype.element_ty), mask=mask)
    tl.store(ghv_ptr + base + offs_d, ghv.to(ghv_ptr.dtype.element_ty), mask=mask)
    tl.store(gxv_ptr + base + offs_d, gxv.to(gxv_ptr.dtype.element_ty), mask=mask)


class FusedGRULTI(torch.autograd.Function):
    @staticmethod
    def forward(ctx, proj_x, proj_h, h, x_encoded, A_scale, B_scale):
        shape = h.shape
        N = shape[0] * shape[1]
        D = shape[2]
        px = proj_x.reshape(N, 3 * D).contiguous()
        ph = proj_h.reshape(N, 3 * D).contiguous()
        hf = h.reshape(N, D).contiguous()
        xf = x_encoded.reshape(N, D).contiguous()
        out = torch.empty((N, D), device=h.device, dtype=h.dtype)
        grid = (N, triton.cdiv(D, 256))
        _gru_lti_fwd_kernel[grid](px, ph, hf, xf, A_scale, B_scale, out, N, D, BLOCK=256)
        ctx.save_for_backward(proj_x, proj_h, h, x_encoded, A_scale, B_scale)
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, grad_out):
        proj_x, proj_h, h, x_encoded, A_scale, B_scale = ctx.saved_tensors
        shape = h.shape
        N = shape[0] * shape[1]
        D = shape[2]
        g = grad_out.reshape(N, D).contiguous()
        px = proj_x.reshape(N, 3 * D).contiguous()
        ph = proj_h.reshape(N, 3 * D).contiguous()
        hf = h.reshape(N, D).contiguous()
        xf = x_encoded.reshape(N, D).contiguous()
        gx = torch.empty_like(px)
        gh = torch.empty_like(ph)
        ghv = torch.empty_like(hf)
        gxv = torch.empty_like(hf)
        grid = (N, triton.cdiv(D, 256))
        _gru_lti_bwd_kernel[grid](px, ph, hf, xf, A_scale, B_scale, g,
                                   gx, gh, ghv, gxv, N, D, BLOCK=256)
        # A/B grads: simple reductions (no cross-N atomics needed)
        xr, xz, xn = proj_x.chunk(3, dim=-1)
        hr, hz, hn = proj_h.chunk(3, dim=-1)
        r = torch.sigmoid(xr + hr)
        z = torch.sigmoid(xz + hz)
        n = torch.tanh(xn + r * hn)
        h_cand = (1 - z) * n + z * h
        gA = (grad_out * h_cand).sum(dim=(0, 1))
        gB = (grad_out * x_encoded).sum(dim=(0, 1))
        return (gx.reshape(shape[0], shape[1], 3 * D),
                gh.reshape(shape[0], shape[1], 3 * D),
                ghv.reshape(shape), gxv.reshape(shape), gA, gB)
