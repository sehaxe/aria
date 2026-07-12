"""HFW-KAN: Hermite-Fourier-Wavelet KAN layer with Gumbel-Softmax router."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .sct import SCTLinear
from kernels.hfw_basis import FusedHFWBasis


class _Fp8KANLinear(torch.autograd.Function):
    # ponytail: amax-divide (x/|x|max -> fp8) + scale=|x|max; scaled_mm multiplies back.
    @staticmethod
    def forward(ctx, x, w, dtype):
        ctx.save_for_backward(x, w)
        sa = x.abs().amax().clamp_min(1e-12).float()
        sw = w.abs().amax().clamp_min(1e-12).float()
        A = (x / sa).to(torch.float8_e4m3fn)
        B = (w.T / sw).contiguous().to(torch.float8_e4m3fn)  # (K, O)
        return torch._scaled_mm(A, B, sa, sw, out_dtype=dtype)

    @staticmethod
    def backward(ctx, grad):
        x, w = ctx.saved_tensors
        return grad @ w, grad.T @ x, None


class HermiteFourierWaveletBasis(nn.Module):
    """Стабильный базис функций Эрмита-Фурье-Вейвлета (HFW-Basis)."""
    def __init__(self, degree=6, num_frequencies=3):
        super().__init__()
        self.degree = degree
        self.num_frequencies = num_frequencies

    def forward(self, x):
        """
        Вход: x (B, S, D)
        Выход: phi (B, S, D, M), M = degree + 2 * num_frequencies
        """
        x_scaled = 2.0 * torch.tanh(x / 2.0)

        # Полиномы Эрмита по рекуррентной схеме
        hermite = []
        h_prev2 = torch.ones_like(x_scaled)
        hermite.append(h_prev2)
        if self.degree > 1:
            h_prev1 = 2.0 * x_scaled
            hermite.append(h_prev1)
            for n in range(2, self.degree):
                h_curr = 2.0 * x_scaled * h_prev1 - 2.0 * (n - 1) * h_prev2
                hermite.append(h_curr)
                h_prev2, h_prev1 = h_prev1, h_curr
        H = torch.stack(hermite, dim=-1)

        # Гауссова огибающая → функции Эрмита (wavelet part)
        envelope = torch.exp(-0.5 * (x_scaled ** 2)).unsqueeze(-1)
        wavelet_part = H * envelope

        # Фурье-гармоники
        if self.num_frequencies > 0:
            fourier = []
            for k in range(1, self.num_frequencies + 1):
                fourier.append(torch.sin(k * x_scaled))
                fourier.append(torch.cos(k * x_scaled))
            fourier_part = torch.stack(fourier, dim=-1)
            phi = torch.cat([wavelet_part, fourier_part], dim=-1)
        else:
            phi = wavelet_part
        return phi


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
        if self.fp8:
            base_out = _Fp8KANLinear.apply(x_flat, self.base_weight, torch.bfloat16)
        else:
            base_out = F.linear(x_flat, self.base_weight)
        base_out = base_out.reshape(*lead, O)
        phi = FusedHFWBasis.apply(x, self.degree, self.num_frequencies)
        B, S, I2, M = phi.shape
        w = self.coeffs.reshape(O, I2 * M)
        phi_flat = phi.reshape(-1, I2 * M)
        if self.fp8:
            kan_out = _Fp8KANLinear.apply(phi_flat, w, torch.bfloat16)
        else:
            kan_out = F.linear(phi_flat, w)
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