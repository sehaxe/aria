import torch
import torch.nn as nn
import math

class DepthLoRA(nn.Module):
    def __init__(self, dim, rank=8, max_loops=6):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(max_loops, dim, rank) * (0.02 / math.sqrt(dim)))
        self.lora_B = nn.Parameter(torch.randn(max_loops, rank, dim) * (0.02 / math.sqrt(rank)))

    def forward(self, x, step):
        A = self.lora_A[step]
        B = self.lora_B[step]
        return (x @ A) @ B
