import torch, torch.nn as nn, math, torch.nn.functional as F
from aria.bitnet_v2 import hadamard, _dequant, _pack_int4

def _ternary_quant(w, ternary=True):
    """Ternary/linear quant of SCT factors with STE.

    Mean-abs normalize, clamp to [-1,1]; ternary hardens to {-1,0,1} (1.58-bit).
    STE: grad flows straight to w. Pure tensor ops → torch.compile-friendly.
    U/V are static within a step (only change after optimizer.step), so this is
    cheap; the caller caches the result keyed on U._version.
    """
    au = w.abs().mean().clamp(min=1e-12)
    ws = w / au
    if ternary:
        q = ws.clamp(-1, 1).round()   # {-1,0,1} via round of clamped (STE through round)
    else:
        q = ws.clamp(-1, 1)
    return (q - w).detach() + w


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
        U, V = self.U, self.V
        s, gamma = self.s, self.gamma
        # Cast U/V to the compute dtype once (cheap); the param version drives the cache key.
        if U.dtype != dtype:
            U = U.to(dtype)
        if V.dtype != dtype:
            V = V.to(dtype)
        # Ternary quant is a pure tensor op (STE) — recompute each call. Cheap,
        # and avoids nn.Parameter._version access which breaks torch.compile
        # (data-dependent guard). Inductor fuses the matmul+quant together.
        u_q = _ternary_quant(U, self.ternary)
        v_q = _ternary_quant(V, self.ternary)
        if s.dtype != dtype:
            s = s.to(dtype)
        if gamma.dtype != dtype:
            gamma = gamma.to(dtype)
        # Parcae: hard-bind spectral norm of injected factors (s, gamma) so the
        # recurrent-core feed-through can't blow up (avoids spectral radius > 1)
        if self.max_sigma > 0:
            sigma = (s * gamma.abs()).amax()
            scale = (sigma / self.max_sigma).clamp(min=1.0)
            s = s / scale
            gamma = gamma / scale
        # ponytail: BitNet v2 — Hadamard + int-quant + matmul. Pure tensor ops
        # (no autograd.Function) so the whole SCTLinear is torch.compile-friendly.
        # STE flows through the quant; Hadamard is orthogonal (H^2=I) so its
        # backward cancels in the residual stream.
        if self.bitnet_v2:
            xq = _bitnet_act_deq(x, self.bitnet_act_bits, self.bitnet_hadamard, dtype)
            return (xq @ u_q * s) @ v_q.T * gamma
        if self.fp8_compute:
            return _fp8_sctmm(x, u_q, s, v_q, gamma)
        if self.use_triton:
            x2 = x.reshape(-1, x.shape[-1]).contiguous()
            out = _sct_mm_ref(x2, u_q, s, v_q, gamma)
            return out.reshape(*x.shape[:-1], out.shape[-1])
        return (x @ u_q * s) @ v_q.T * gamma


def _bitnet_act_deq(x, bits, use_hadamard, dtype):
    """BitNet activation quant → dequantized float (compile-friendly, STE)."""
    from aria.bitnet_v2 import _bitnet_act_quant
    return _bitnet_act_quant(x, bits, use_hadamard).to(dtype)


def _fp8_sctmm(x, u, s, v, gamma):
    """e4m3 scaled-mm SCT (compile-friendly; needs sm_89+)."""
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    pad = (16 - u.shape[-1] % 16) % 16
    uu, vv, ss = u, v, s
    if pad:
        uu = F.pad(u, (0, pad)); vv = F.pad(v, (0, pad)); ss = F.pad(s, (0, pad))
    sa = (x2.abs().amax() + 1e-12).float()
    su = (uu.abs().amax() + 1e-12).float()
    sv = (vv.abs().amax() + 1e-12).float()
    h = torch._scaled_mm((x2 / sa).to(torch.float8_e4m3fn),
                         (uu / su).to(torch.float8_e4m3fn), sa, su, out_dtype=x.dtype)
    h = h * ss
    sh = (h.abs().amax() + 1e-12).float()
    out = torch._scaled_mm((h / sh).to(torch.float8_e4m3fn),
                           (vv / sv).to(torch.float8_e4m3fn).T, sh, sv, out_dtype=x.dtype)
    return out.reshape(*x.shape[:-1], x.shape[-1]) * gamma


def _sct_mm_ref(x, u, s, v, gamma):
    """Reference low-rank matmul (compile-friendly; inductor fuses it)."""
    return (x @ u * s) @ v.T * gamma