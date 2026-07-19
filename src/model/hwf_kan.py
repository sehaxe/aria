"""HFW-KAN: Hermite-Fourier-Wavelet KAN layer with Gumbel-Softmax router."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .sct import SCTLinear


def hfw_basis(x, degree=6, num_frequencies=3):
    """Hermite-Fourier-Wavelet basis, pure tensor ops (torch.compile-friendly).

    Input x (B,S,D) -> phi (B,S,D,M), M = degree + 2*num_frequencies.
    Replaces the Triton FusedHFWBasis (autograd.Function) so the whole KAN
    layer fuses under torch.compile (fullgraph).
    """
    x_scaled = 2.0 * torch.tanh(x / 2.0)
    hermite = [torch.ones_like(x_scaled)]
    if degree > 1:
        h_prev2, h_prev1 = hermite[0], 2.0 * x_scaled
        hermite.append(h_prev1)
        for n in range(2, degree):
            h_curr = 2.0 * x_scaled * h_prev1 - 2.0 * (n - 1) * h_prev2
            hermite.append(h_curr)
            h_prev2, h_prev1 = h_prev1, h_curr
    H = torch.stack(hermite, dim=-1)
    wavelet_part = H * torch.exp(-0.5 * (x_scaled * x_scaled)).unsqueeze(-1)
    if num_frequencies > 0:
        fourier = [comp(x_scaled) for k in range(1, num_frequencies + 1)
                   for comp in (torch.sin, torch.cos)]
        phi = torch.cat([wavelet_part, torch.stack(fourier, dim=-1)], dim=-1)
    else:
        phi = wavelet_part
    return phi


def _fp8_linear(x, w, dtype):
    """e4m3 scaled-mm (compile-friendly; no autograd.Function)."""
    sa = x.abs().amax().clamp_min(1e-12).float()
    sw = w.abs().amax().clamp_min(1e-12).float()
    A = (x / sa).to(torch.float8_e4m3fn)
    B = (w.T / sw).contiguous().to(torch.float8_e4m3fn)
    return torch._scaled_mm(A, B, sa, sw, out_dtype=dtype)


class HFW_KANLayer(nn.Module):
    """Полностью векторизованный HFW-KAN слой."""
    def __init__(self, in_features, out_features, degree=6, num_frequencies=3,
                 max_sigma=1.0, fp8=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.num_frequencies = num_frequencies
        self.num_bases = degree + 2 * num_frequencies
        self.max_sigma = max_sigma
        self.fp8 = fp8

        self.base_weight = nn.Parameter(
            torch.randn(out_features, in_features) * (1.0 / math.sqrt(in_features))
        )
        self.coeffs = nn.Parameter(
            torch.randn(out_features, in_features, self.num_bases) *
            (0.02 / math.sqrt(in_features * self.num_bases))
        )

    def forward(self, x):
        *lead, I = x.shape
        O = self.out_features
        x_flat = x.reshape(-1, I)
        base_out = _fp8_linear(x_flat, self.base_weight, torch.bfloat16) if self.fp8 \
            else F.linear(x_flat, self.base_weight)
        base_out = base_out.reshape(*lead, O)
        phi = hfw_basis(x, self.degree, self.num_frequencies)
        B, S, I2, M = phi.shape
        w = self.coeffs.reshape(O, I2 * M)
        phi_flat = phi.reshape(-1, I2 * M)
        kan_out = _fp8_linear(phi_flat, w, torch.bfloat16) if self.fp8 \
            else F.linear(phi_flat, w)
        kan_out = kan_out.reshape(*lead, O)
        return base_out + kan_out


class SwiGLU(nn.Module):
    """Спектрально-стабильный SwiGLU на базе SCTLinear."""
    def __init__(self, dim, hidden_dim, rank=32, max_sigma=1.0):
        super().__init__()
        self.w1 = SCTLinear(dim, hidden_dim, rank=rank, max_sigma=max_sigma)
        self.w2 = SCTLinear(dim, hidden_dim, rank=rank, max_sigma=max_sigma)
        self.w3 = SCTLinear(hidden_dim, dim, rank=rank, max_sigma=max_sigma)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class DynamicFFN(nn.Module):
    def __init__(self, dim, degree=6, num_frequencies=3, temperature=1.0, rank=32,
                 max_sigma=1.0, fp8_kan=False):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        hidden_dim = int(8 * dim / 3)
        self.mlp_expert = SwiGLU(dim, hidden_dim, rank=rank, max_sigma=max_sigma)
        self.kan_expert = HFW_KANLayer(dim, dim, degree, num_frequencies,
                                       max_sigma=max_sigma, fp8=fp8_kan)
        self.router = nn.Linear(dim, 2)
        self.register_buffer("h_suppression_factor", torch.tensor(0.5))

    def forward(self, x, confidence=None):
        route_logits = self.router(x)
        if self.training:
            weights = F.gumbel_softmax(route_logits, tau=self.temperature, hard=False, dim=-1)
        else:
            weights = F.softmax(route_logits, dim=-1)
        mlp_out = self.mlp_expert(x)
        kan_out = self.kan_expert(x)
        if confidence is not None:
            damping = self.h_suppression_factor + (1.0 - self.h_suppression_factor) * confidence
            mlp_out = mlp_out * damping
        output = weights[..., 0:1] * mlp_out + weights[..., 1:2] * kan_out
        return output, route_logits