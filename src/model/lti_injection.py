import torch
import torch.nn as nn


class LTIInjection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # ponytail: mHC manifold A+B=1.0 via sigmoid gate (sigmoid(3.0)~0.95).
        self.gate_param = nn.Parameter(torch.ones(dim) * 3.0)

    @property
    def A_scale(self):
        return torch.sigmoid(self.gate_param)

    @property
    def B_scale(self):
        return 1.0 - self.A_scale

    def forward(self, h_candidate, x_encoded):
        return self.A_scale * h_candidate + self.B_scale * x_encoded
