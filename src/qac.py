import torch

def quantize(x, bits=8):
    if bits == 0:
        return x
    abs_max = x.abs().max().clamp(min=1e-12)
    if bits >= 8:
        qmax = 448.0
    else:
        qmax = 6.0
    scale = qmax / abs_max
    dither = torch.rand_like(x) - 0.5
    q = (x * scale + dither).floor()
    return x + (q / scale - x).detach()
