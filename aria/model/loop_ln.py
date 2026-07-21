import torch
import torch.nn as nn
import torch.nn.functional as F


class LoopModulatedLN(nn.Module):
    def __init__(self, dim, max_loops=6):
        super().__init__()
        self.dim = dim
        self.max_loops = max_loops
        self.weight = nn.Parameter(torch.ones(max_loops, dim))
        self.bias = nn.Parameter(torch.zeros(max_loops, dim))

    def forward(self, x, step):
        return F.layer_norm(x, (self.dim,), self.weight[step], self.bias[step])
