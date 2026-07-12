import torch, torch.nn as nn, torch.nn.functional as F
from .sct import SCTLinear

class WaveletKANFFN(nn.Module):
    """Wav-KAN: gate * psi where psi = (1-u²)exp(-0.5u²)."""
    def __init__(self, d_model, d_ffn=None, rank=32, max_sigma=1.0):
        super().__init__()
        d = d_ffn or d_model
        self.gate = SCTLinear(d_model, d, rank, max_sigma=max_sigma)
        self.up = SCTLinear(d_model, d, rank, max_sigma=max_sigma)
        self.down = SCTLinear(d, d_model, rank, max_sigma=max_sigma)
        self.log_scale = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        g = F.silu(self.gate(x))
        u = torch.tanh(self.up(x)) * self.log_scale.exp().view(1, 1, -1)
        psi = (1.0 - u * u) * torch.exp(-0.5 * u * u)
        return self.down(g * psi)
