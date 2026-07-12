import torch

def cld_logits(logits_deep, logits_shallow, gamma=0.1):
    return tuple(d - s * gamma for d, s in zip(logits_deep, logits_shallow))
