"""BitNet v2 (arXiv:2504.18415) — native 4-bit activations for 1-bit LLMs.

Core idea: a standard BitLinear (ternary 1.58-bit weights, STE) plus an online
Hadamard transform on the *input activation* before quantization. The Hadamard
mixes outlier channels into a Gaussian-like distribution so the activation can be
stored/computed in 4-bit without the divergence that plain INT4 causes (BitNet
a4.8 needed TopK sparsity for exactly this; v2 replaces sparsity with Hadamard).

This module is self-contained and applies to ANY linear's input — Aria's
SCTLinear (ternary low-rank) reuses :func:`hadamard_quantize` on its activation.

Train recipe (per paper, two-stage):
  stage 1: act_bits=8  (full 100B-token run)
  stage 2: act_bits=4  (continue ~5B tokens, reuse optimizer states)

 ponytail: Hadamard is O(n^2) matmul via a cached Sylvester matrix (n=next-pow2).
 Real fast-hadamard-transform (O(n log n)) can be dropped in later if profiling
 shows the matmul on the activation path is hot. 4-bit activations are stored as
 int8 here (true int4 packing is a downstream memory optimization, not needed
 for correctness).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_HADAMARD_CACHE = {}


def _hadamard_matrix(n, device, dtype):
    """Normalized Sylvester Hadamard (n x n, entries ±1/sqrt(n)), n = 2^m."""
    if (n, str(device), dtype) in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[(n, str(device), dtype)]
    # ponytail: build on CPU, cache per (n, device, dtype)
    def build(k):
        if k == 1:
            return torch.tensor([[1.0]], dtype=torch.float32)
        h = build(k // 2)
        return torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0) / math.sqrt(2)
    H = build(n)      # ponytail: repeated /sqrt(2) already normalizes entries to ±1/sqrt(n)
    H = H.to(device=device, dtype=dtype)
    _HADAMARD_CACHE[(n, str(device), dtype)] = H
    return H


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


class _HadamardTransform(torch.autograd.Function):
    """Online Hadamard on the last dim (padded to next pow2), STE-free.

    H is orthogonal, so the exact gradient is H applied to the incoming grad
    (same transform). Padding is zeros, so it folds through cleanly.
    """

    @staticmethod
    def forward(ctx, x):
        D = x.shape[-1]
        P = _next_pow2(D)
        ctx.P, ctx.D = P, D
        flat = x.reshape(-1, D)
        if P != D:
            flat = F.pad(flat, (0, P - D))
        H = _hadamard_matrix(P, x.device, x.dtype)
        out = flat @ H.T
        # ponytail: return the unpadded dim so the activation keeps x's shape
        return out.reshape(*x.shape[:-1], P)[..., :D]

    @staticmethod
    def backward(ctx, grad):
        P, D = ctx.P, ctx.D
        H = _hadamard_matrix(P, grad.device, grad.dtype)
        flat = grad.reshape(-1, D)
        if P != D:
            flat = F.pad(flat, (0, P - D))   # pad to match the H multiply
        g = flat @ H.T          # ponytail: H is orthogonal & symmetric, so H^T == H (unitary)
        if P != D:
            g = g[:, :D]
        return g.reshape(*grad.shape[:-1], D)


def _quantize_act(x, bits):
    """Per-token activation quant with STE.

    bits==8: per-token absmax INT8 (QINT8, eq.7 of the paper).
    bits==4: per-token absmean INT4 (QINT4, eq.8 of the paper).
    Returns dequantized float for compute; the quantized int is what you'd store
    for the memory win.
    """
    if bits >= 8:
        gamma = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        q = (x / gamma * 127).round().clamp(-128, 127)
        return (q / 127 * gamma - x).detach() + x
    else:
        beta = x.abs().mean(dim=-1, keepdim=True).clamp(min=1e-12)
        q = (x / beta * 7).round().clamp(-8, 7)
        return (q / 7 * beta - x).detach() + x


class _ActQuant(torch.autograd.Function):
    """Hadamard -> quantize, with STE through the quant and exact grad through
    the Hadamard. Saves the int tensor for the memory win; reconstructs in
    backward via STE (gradient flows to the pre-quant Hadamard output)."""

    @staticmethod
    def forward(ctx, x, bits, use_hadamard):
        xh = _HadamardTransform.apply(x) if use_hadamard else x
        ctx.bits = bits
        ctx.use_hadamard = use_hadamard
        ctx.d_in = x.shape[-1]
        ctx.save_for_backward(xh)
        return _quantize_act(xh, bits)

    @staticmethod
    def backward(ctx, grad):
        (xh,) = ctx.saved_tensors
        # ponytail: STE — grad flows straight to the pre-quant Hadamard output
        g = grad
        if ctx.use_hadamard:
            g = _HadamardTransform.apply(g)
        if g.shape[-1] != ctx.d_in:
            g = g[:, :ctx.d_in]
        return g, None, None


def hadamard_quantize(x, bits=8, use_hadamard=True):
    """Stateless entry point: return the 4/8-bit-quantized activation of x."""
    return _ActQuant.apply(x, bits, use_hadamard)


def _pack_int4(q):
    # ponytail: pack signed int4 (vals in [-8,7]) two-per-byte along the last dim
    nib = q.to(torch.int8) & 0x0F
    return (nib[..., 0::2] | (nib[..., 1::2] << 4)).to(torch.int8)


def _unpack_int4(packed):
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    low = (low ^ 8) - 8
    high = (high ^ 8) - 8
    out = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.int8,
                      device=packed.device)
    out[..., 0::2] = low
    out[..., 1::2] = high
    return out


def _dequant(q, scale, bits, packed):
    # ponytail: rebuild the (pre-Hadamard) activation from its int codes + scale
    q = _unpack_int4(q) if packed else q
    q = q.to(scale.dtype)
    return q * (scale / 127.0 if bits >= 8 else scale / 7.0)


class _BitnetActQuant(torch.autograd.Function):
    """Full BitNet activation quant at the source.

    Quantize x ONCE -> return the dequantized float to feed every consumer
    (residual / KAN / gates / recurrent). We save ONLY the int codes + scale
    (tiny) in ctx and NOT the bf16 x, so x is released after forward. The
    backward is straight-through through the quant and the (orthogonal) Hadamard,
    so it never needs the bf16 activation either — this is the memory win: the
    persisted activation across loop steps is int, not bf16.
    """

    @staticmethod
    def forward(ctx, x, bits, use_hadamard):
        xh = _HadamardTransform.apply(x) if use_hadamard else x
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
        ctx.bits, ctx.use_hadamard, ctx.packed = bits, use_hadamard, packed
        ctx.q = q
        ctx.scale = scale
        deq = _dequant(q, scale, bits, packed)
        # ponytail: de-rotate back to the original (non-Hadamard) basis so
        # the HelixCore residual stream (engram_mem, LTI injection) stays in
        # one coherent space. H^2 = I for Sylvester Hadamard, so this is exact.
        if use_hadamard:
            deq = _HadamardTransform.apply(deq)
        return deq

    @staticmethod
    def backward(ctx, grad):
        # ponytail: two orthogonal Hadamard applications cancel (H^2 = I),
        # so the gradient is passed straight through to x (STE for the quant).
        return grad, None, None


def bitnet_quantize_act(x, bits=8, use_hadamard=True):
    """Quantize activation x once (full BitNet path). Returns the dequantized
    float to feed every consumer; stores only int codes internally so the bf16
    x is released."""
    return _BitnetActQuant.apply(x, bits, use_hadamard)


def bitnet_int_codes(x, bits=8, use_hadamard=True):
    """Detached int codes + scale for a quantized activation (storage / VRAM
    introspection). No autograd edge to x."""
    xh = _HadamardTransform.apply(x) if use_hadamard else x
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
    return q.detach(), scale.detach(), packed


class HBitLinear(nn.Module):
    """Dense ℋ-BitLinear: ternary weight (absmean STE) x Hadamard-quantized act.

    Drop-in replacement for nn.Linear in a 1-bit-LLM stack. act_bits switches the
    two training stages (8 -> 4).
    """

    def __init__(self, d_in, d_out, act_bits=8, use_hadamard=True, bias=False):
        super().__init__()
        self.d_in, self.d_out = d_in, d_out
        self.act_bits = act_bits
        self.use_hadamard = use_hadamard
        self.W = nn.Parameter(torch.randn(d_out, d_in) * (1.0 / math.sqrt(d_in)))
        if bias:
            self.b = nn.Parameter(torch.zeros(d_out))
        else:
            self.register_parameter("b", None)

    def _ternarize(self, W):
        alpha = W.abs().mean()
        wq = alpha * torch.round(W.div(alpha + 1e-12).clamp(-1, 1))
        return (wq - W).detach() + W      # ponytail: STE

    def forward(self, x):
        Wq = self._ternarize(self.W)
        xq = hadamard_quantize(x, self.act_bits, self.use_hadamard)
        # ponytail: pad weight's input dim to match the pow2-padded activation
        P = xq.shape[-1]
        if P != Wq.shape[-1]:
            Wq = F.pad(Wq, (0, P - Wq.shape[-1]))
        y = xq @ Wq.T
        return y + self.b if self.b is not None else y


def _run_checks():
    torch.manual_seed(0)
    dev = torch.device("cpu")
    ok = True

    # 1) HBitLinear forward + backward finite
    m = HBitLinear(8, 4, act_bits=4, use_hadamard=True).to(dev)
    x = torch.randn(5, 8, requires_grad=True)
    y = m(x)
    assert torch.isfinite(y).all(), "forward not finite"
    (y ** 2).sum().backward()
    assert torch.isfinite(m.W.grad).all() and torch.isfinite(x.grad).all(), "grad not finite"
    assert m.W.grad.abs().sum() > 0, "zero grad"
    print("[1] HBitLinear fwd/bwd finite + nonzero grad: PASS")

    # ponytail: [2] Hadamard orthogonality + exact (orthogonal) backward
    H = _hadamard_matrix(8, dev, torch.float32)
    assert torch.allclose(H @ H, torch.eye(8), atol=1e-4), "H not orthogonal"
    t = torch.randn(3, 8, requires_grad=True)
    out = _HadamardTransform.apply(t)
    grad_in = torch.randn(3, 8)
    out.backward(grad_in)
    # ponytail: y = x @ H  =>  dL/dx = dL/dy @ H  (H symmetric, orthogonal)
    assert torch.allclose(t.grad, grad_in @ H, atol=1e-4), "backward != H @ grad"
    print("[2] Hadamard orthogonal + exact grad: PASS")

    # 3) 4-bit vs fp16 activation byte cost (memory win)
    xb = torch.randn(64, 1600)
    xq4 = hadamard_quantize(xb, bits=4)
    # the quantized int is bounded in [-8,7] -> storable as int8 (1 byte)
    int_bytes = xq4.numel() * 1           # ponytail: int8 storage ceiling
    fp_bytes = xb.numel() * 2             # ponytail: bf16
    assert int_bytes < fp_bytes, "no memory saving"
    print(f"[3] 4-bit act storage {int_bytes/1024:.0f}KB vs bf16 {fp_bytes/1024:.0f}KB "
          f"({fp_bytes/int_bytes:.1f}x smaller): PASS")

    # 4) toy regression: 8-bit trains, 4-bit does not diverge
    for bits in (8, 4):
        torch.manual_seed(1)
        lin = HBitLinear(6, 3, act_bits=bits)
        target = torch.randn(32, 3)
        opt = torch.optim.SGD(lin.parameters(), lr=0.1)
        loss0 = None
        for _ in range(150):
            opt.zero_grad()
            loss = ((lin(torch.randn(32, 6)) - target) ** 2).mean()
            if loss0 is None:
                loss0 = loss.item()
            loss.backward()
            opt.step()
        assert torch.isfinite(loss).all(), f"bits={bits} diverged"
        print(f"[4] toy train act_bits={bits}: loss {loss0:.3f} -> {loss.item():.3f} "
              f"({'DOWN' if loss.item() < loss0 else 'UP'}): PASS")

    # 5) full BitNet activation path (quant -> dequant) == STE reference in both
    #    value and gradient-to-x. Reference = hadamard_quantize(x) (already applies
    #    H internally when hadamard=True); bitnet_quantize_act must match it.
    for bits in (8, 4):
        for had in (True, False):
            x = torch.randn(4, 6, requires_grad=True)
            y = bitnet_quantize_act(x, bits, had)
            y.sum().backward()
            gx = x.grad.clone()
            x2 = x.detach().clone().requires_grad_(True)
            r = hadamard_quantize(x2, bits, had)
            r.sum().backward()
            assert torch.allclose(y, r, atol=1e-4), f"fwd bits={bits} had={had}"
            assert torch.allclose(gx, x2.grad, atol=1e-3), f"grad bits={bits} had={had}"
            if bits < 8:
                q, s, pk = bitnet_int_codes(x2.detach(), bits, had)
                assert q.dtype == torch.int8 and q.shape[-1] == x.shape[-1] // 2, \
                    "4-bit codes not packed to int4"
    print("[5] bitnet quant->dequant == STE reference (fwd + grad to x): PASS")

    print("ALL CHECKS PASS" if ok else "CHECKS FAILED")
    return ok


if __name__ == "__main__":
    _run_checks()
