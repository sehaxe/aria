import torch
import triton
import triton.language as tl

# Max basis dimension across configs: degree<=8, num_freqs<=4 -> M<=16 (power-of-2 for arange)
MAX_M = 16


@triton.jit
def _hfw_basis_fwd(
    x_ptr, out_ptr,
    N, D, M: tl.constexpr, MAX_M: tl.constexpr, degree: tl.constexpr, num_freqs: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N * D
    x_val = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    e2 = tl.exp(x_val)
    tanh_half = (e2 - 1.0) / (e2 + 1.0)
    x_scaled = 2.0 * tanh_half
    envelope = tl.exp(-0.5 * x_scaled * x_scaled)

    col = out_ptr + offsets * MAX_M
    h_prev2 = tl.full((BLOCK,), 1.0, dtype=tl.float32)
    h_prev1 = 2.0 * x_scaled
    tl.store(col + 0, h_prev2 * envelope, mask=mask)
    tl.store(col + 1, h_prev1 * envelope, mask=mask)
    h2 = h_prev2
    h1 = h_prev1
    for n in range(2, degree):
        h_curr = 2.0 * x_scaled * h1 - 2.0 * (n - 1) * h2
        tl.store(col + n, h_curr * envelope, mask=mask)
        h2 = h1
        h1 = h_curr

    for k in range(1, num_freqs + 1):
        kf = k * x_scaled
        tl.store(col + (degree + 2 * (k - 1)), tl.sin(kf), mask=mask)
        tl.store(col + (degree + 2 * (k - 1) + 1), tl.cos(kf), mask=mask)


@triton.jit
def _hfw_basis_bwd(
    x_ptr, grad_out_ptr, grad_x_ptr,
    N, D, M: tl.constexpr, MAX_M: tl.constexpr, degree: tl.constexpr, num_freqs: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N * D
    x_val = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    e2 = tl.exp(x_val)
    tanh_half = (e2 - 1.0) / (e2 + 1.0)
    x_scaled = 2.0 * tanh_half
    dx_scaled = 1.0 - tanh_half * tanh_half
    envelope = tl.exp(-0.5 * x_scaled * x_scaled)
    d_env = -x_scaled * envelope

    gcol = grad_out_ptr + offsets * M
    grad_x = tl.zeros((BLOCK,), dtype=tl.float32)

    h2 = tl.full((BLOCK,), 1.0, dtype=tl.float32)
    h1 = 2.0 * x_scaled
    dh2 = tl.full((BLOCK,), 0.0, dtype=tl.float32)
    dh1 = tl.full((BLOCK,), 2.0, dtype=tl.float32)
    g0 = tl.load(gcol + 0, mask=mask)
    g1 = tl.load(gcol + 1, mask=mask)
    grad_x += g0 * ((dh2 * envelope + h2 * d_env) * dx_scaled)
    grad_x += g1 * ((dh1 * envelope + h1 * d_env) * dx_scaled)
    for n in range(2, degree):
        h_curr = 2.0 * x_scaled * h1 - 2.0 * (n - 1) * h2
        dh_curr = 2.0 * n * h1
        g_n = tl.load(gcol + n, mask=mask)
        grad_x += g_n * ((dh_curr * envelope + h_curr * d_env) * dx_scaled)
        h2 = h1
        h1 = h_curr
        dh2 = dh1
        dh1 = dh_curr

    for k in range(1, num_freqs + 1):
        kf = k * x_scaled
        g_sin = tl.load(gcol + (degree + 2 * (k - 1)), mask=mask)
        g_cos = tl.load(gcol + (degree + 2 * (k - 1) + 1), mask=mask)
        grad_x += g_sin * (k * tl.cos(kf) * dx_scaled)
        grad_x += g_cos * (-k * tl.sin(kf) * dx_scaled)

    tl.store(grad_x_ptr + offsets, grad_x, mask=mask)


class FusedHFWBasis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, degree=6, num_freqs=3):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        N, D = flat.shape
        M = degree + 2 * num_freqs
        out = torch.empty((N, D, MAX_M), device=x.device, dtype=torch.float32)
        BLOCK = 1024
        grid = (triton.cdiv(N * D, BLOCK),)
        _hfw_basis_fwd[grid](flat, out, N, D, M, MAX_M, degree, num_freqs, BLOCK)
        ctx.save_for_backward(flat)
        ctx.degree = degree
        ctx.num_freqs = num_freqs
        ctx.input_shape = shape
        return out[:, :, :M].reshape(*shape, M).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        flat = x.reshape(-1, x.shape[-1])
        N, D = flat.shape
        M = ctx.degree + 2 * ctx.num_freqs
        grad_x = torch.empty_like(flat, dtype=torch.float32)
        go = grad_out.reshape(-1, D, M).to(torch.float32)
        BLOCK = 1024
        grid = (triton.cdiv(N * D, BLOCK),)
        _hfw_basis_bwd[grid](flat, go, grad_x, N, D, M, MAX_M, ctx.degree, ctx.num_freqs, BLOCK)
        return grad_x.reshape(ctx.input_shape).to(x.dtype), None, None
