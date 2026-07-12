import torch, torch.nn as nn, math, torch.nn.functional as F
from kernels.sct_quant import FusedSCTQuant
from kernels.sct_mm import _sct_fwd_kernel
import triton
import triton.language as tl
from bitnet_v2 import _HadamardTransform, _dequant, _pack_int4

class _SCTMM(torch.autograd.Function):
    """Low-rank matmul (x@u*s)@v^T*gamma via the fused Triton kernel (fwd) +
    analytic backward. The kernel is fp32; STE quantization of u/v flows in
    separately through FusedSCTQuant, so this only owns the matmul gradients.
    """
    @staticmethod
    def forward(ctx, x, u, s, v, gamma):
        ctx.save_for_backward(x, u, s, v, gamma)
        M, K = x.shape
        R = u.shape[1]
        N = v.shape[0]
        out = torch.empty(M, N, dtype=x.dtype, device=x.device)
        B_M = B_N = B_K = B_R = 32
        grid = (triton.cdiv(M, B_M), triton.cdiv(N, B_N))
        out_dtype = tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float32
        _sct_fwd_kernel[grid](x, u, s, v, out, M, K, N, R,
                               x.stride(0), x.stride(1),
                               u.stride(0), u.stride(1),
                               v.stride(0), v.stride(1),
                               out.stride(0), out.stride(1),
                               B_M, B_N, B_K, B_R, out_dtype)
        return out * gamma

    @staticmethod
    def backward(ctx, grad_out):
        x, u, s, v, gamma = ctx.saved_tensors
        g = grad_out * gamma                       # dL/dz, z = ((x@u)*s)@v^T
        h = x @ u
        hs = h * s
        dz = grad_out * gamma
        z = (hs @ v.T)
        dx = (g @ v) @ u.T                          # = dhs@u.T, dhs = g@v
        du = x.T @ (g @ v) * s                       # = x.T @ dh, dh = dhs*s
        ds = ((g @ v) * h).sum(0)
        dv = g.T @ hs
        dgamma = (grad_out * z).sum()
        return dx, du, ds, dv, dgamma


class _BitnetSCTMM(torch.autograd.Function):
    """SCTLinear matmul on the BitNet-quantized activation.

    Stores ONLY the int activation codes (+ tiny scale) in ctx — never the bf16
    activation — so the per-step activation memory is int, not bf16. Backward
    recomputes the dequant from the int codes (STE through the quant, exact
    through the orthogonal Hadamard) and runs the analytic SCTMM backward on it.
    This is the real VRAM win over the plain path (which save_for_backward's x).
    """

    @staticmethod
    def forward(ctx, x, u, s, v, gamma, bits, hadamard):
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        ctx.orig = x.shape[:-1]
        xh = _HadamardTransform.apply(x2) if hadamard else x2
        if bits >= 8:
            scale = xh.abs().amax(-1, keepdim=True).clamp(min=1e-12)
            q = (xh / scale * 127).round().clamp(-128, 127).to(torch.int8)
            packed = False
        else:
            scale = xh.abs().mean(-1, keepdim=True).clamp(min=1e-12)
            q = (xh / scale * 7).round().clamp(-8, 7).to(torch.int8)
            packed = xh.shape[-1] % 2 == 0
            if packed:
                q = _pack_int4(q)           # ponytail: true int4 (2 codes/byte)
        ctx.bits, ctx.hadamard, ctx.packed = bits, hadamard, packed
        ctx.save_for_backward(u, s, v, gamma, q, scale)
        deq = _dequant(q, scale, bits, packed).to(u.dtype)
        M, K = deq.shape
        R = u.shape[1]
        N = v.shape[0]
        out = torch.empty(M, N, dtype=deq.dtype, device=deq.device)
        B_M = B_N = B_K = B_R = 32
        grid = (triton.cdiv(M, B_M), triton.cdiv(N, B_N))
        out_dtype = tl.bfloat16 if deq.dtype == torch.bfloat16 else tl.float32
        _sct_fwd_kernel[grid](deq, u, s, v, out, M, K, N, R,
                              deq.stride(0), deq.stride(1),
                              u.stride(0), u.stride(1),
                              v.stride(0), v.stride(1),
                              out.stride(0), out.stride(1),
                              B_M, B_N, B_K, B_R, out_dtype)
        return out.reshape(*ctx.orig, out.shape[-1]) * gamma

    @staticmethod
    def backward(ctx, grad_out):
        u, s, v, gamma, q, scale = ctx.saved_tensors
        deq = _dequant(q, scale, ctx.bits, ctx.packed).to(u.dtype)
        g = grad_out.reshape(-1, grad_out.shape[-1]) * gamma
        h = deq @ u
        hs = h * s
        z = (hs @ v.T)
        dx = (g @ v) @ u.T
        du = deq.T @ (g @ v) * s
        ds = ((g @ v) * h).sum(0)
        dv = g.T @ hs
        dgamma = (grad_out.reshape(-1, grad_out.shape[-1]) * z).sum()
        # ponytail: STE grad to x (recompute deq in fwd was via int; backward
        # reflects it: grad flows straight to x, exact through the Hadamard).
        if ctx.hadamard:
            dx = _HadamardTransform.apply(dx)
        return dx.reshape(*ctx.orig, dx.shape[-1]), du, ds, dv, dgamma, None, None


class SCTLinear(nn.Module):
    def __init__(self, d_in, d_out, rank=32, ternary=True, fp8=False, max_sigma=1.0,
                 sct_kernel=False, sct_fp8=False, bitnet_v2=False,
                 bitnet_act_bits=8, bitnet_hadamard=True):
        super().__init__()
        self.d_in, self.d_out, self.rank = d_in, d_out, rank
        self.ternary, self.fp8 = ternary, fp8
        self.max_sigma = max_sigma
        self.use_triton = sct_kernel
        self.fp8_compute = sct_fp8 or fp8
        self.bitnet_v2 = bitnet_v2
        self.bitnet_act_bits = bitnet_act_bits
        self.bitnet_hadamard = bitnet_hadamard
        # s init scaled so output std matches a standard nn.Linear (uniform init).
        # Raw ternary u_q/v_q (±1, ~0.69 density) make the effective weight ~6.8x
        # hotter than nn.Linear otherwise — fine behind softmax/GRU gates, but it
        # blows up any SCTLinear on a direct logit path (e.g. SynthesisBlock).
        s = 0.146 / math.sqrt(d_in)
        self.U = nn.Parameter(torch.randn(d_in, rank) * s)
        self.s = nn.Parameter(torch.ones(rank) * s)
        self.V = nn.Parameter(torch.randn(d_out, rank) * s)
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        dtype = x.dtype
        U, V = self.U.to(dtype), self.V.to(dtype)
        s = self.s.to(dtype)
        gamma = self.gamma.to(dtype)
        # Parcae: hard-bind spectral norm of injected factors (s, gamma) so the
        # recurrent-core feed-through can't blow up (avoids spectral radius > 1)
        if self.max_sigma > 0:
            sigma = (s * gamma.abs()).amax()
            scale = (sigma / self.max_sigma).clamp(min=1.0)
            s = s / scale
            gamma = gamma / scale
        # ponytail: monolithic Triton quant replaces clamp + 2x where + one/zero
        # tensor allocs; keeps the same au-mean normalization + STE grad.
        u_q = FusedSCTQuant.apply(U, self.ternary)
        v_q = FusedSCTQuant.apply(V, self.ternary)
        # ponytail: BitNet v2 — Hadamard + int-quant + matmul in one Function that
        # stores only the int activation codes (recomputes dequant in backward) ->
        # real VRAM win over the plain path (which save_for_backward's bf16 x).
        if self.bitnet_v2:
            return _BitnetSCTMM.apply(x, u_q, s, v_q, gamma,
                                      self.bitnet_act_bits, self.bitnet_hadamard)
        if self.fp8_compute:
            return _Fp8SCTMM.apply(x, u_q, s, v_q, gamma)
        if self.use_triton:
            x2 = x.reshape(-1, x.shape[-1]).contiguous()
            out = _SCTMM.apply(x2, u_q, s, v_q, gamma)
            return out.reshape(*x.shape[:-1], out.shape[-1])
        return (x @ u_q * s) @ v_q.T * gamma

class _Fp8SCTMM(torch.autograd.Function):
    # ponytail: e4m3 only; fp4 (e2m1) is the next step — needs scaled-mm support.
    @staticmethod
    def forward(ctx, x, u, s, v, gamma):
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        ctx.save_for_backward(x2, u, s, v, gamma)
        ctx.orig = x.shape[:-1]
        # ponytail: amax-divide scaling (x/|x|max -> fp8) with scale=|x|max passed
        # to scaled_mm, which multiplies back. The reciprocal-prescale form
        # double-applies the scale and is wrong.
        pad = (16 - u.shape[-1] % 16) % 16
        uu, vv, ss = u, v, s
        if pad:
            uu = F.pad(u, (0, pad))
            vv = F.pad(v, (0, pad))
            ss = F.pad(s, (0, pad))
        sa = (x2.abs().amax() + 1e-12).float()
        su = (uu.abs().amax() + 1e-12).float()
        sv = (vv.abs().amax() + 1e-12).float()
        h = torch._scaled_mm((x2 / sa).to(torch.float8_e4m3fn),
                             (uu / su).to(torch.float8_e4m3fn), sa, su, out_dtype=x.dtype)
        h = h * ss
        sh = (h.abs().amax() + 1e-12).float()
        out = torch._scaled_mm((h / sh).to(torch.float8_e4m3fn),
                               (vv / sv).to(torch.float8_e4m3fn).T, sh, sv, out_dtype=x.dtype)
        return out.reshape(*ctx.orig, out.shape[-1]) * gamma

    @staticmethod
    def backward(ctx, grad_out):
        x, u, s, v, gamma = ctx.saved_tensors
        g = grad_out.reshape(-1, grad_out.shape[-1]) * gamma   # dL/dz, z = ((x@u)*s)@v^T
        h = x @ u
        hs = h * s
        dx = (g @ v) @ u.T                          # = dhs@u.T, dhs = g@v
        du = x.T @ (g @ v) * s                       # = x.T @ dh, dh = dhs*s
        ds = ((g @ v) * h).sum(0)
        dv = g.T @ hs
        dgamma = (grad_out.reshape(-1, grad_out.shape[-1]) * (hs @ v.T)).sum()
        return dx.reshape(*ctx.orig, dx.shape[-1]), du, ds, dv, dgamma