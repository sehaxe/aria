"""Integration test for the 5 audit bugs + end-to-end loss training.

Covers, on tiny 29m-style flags (sct_kernel, bitnet_v2, nsa, jepa, mtp,
adaptive_loops):
  * Bug1 (SCT dx *s)        -> SCT grad test (test_sct_grad.py) + grads finite here
  * Bug2 (jepa_dropout)     -> joint CE+JEPA forward/backward no AttributeError
  * Bug3 (bitnet+JEPA torch.stack) -> joint fwd/bwd under bitnet_v2
  * Bug4 (empty-mask JEPA)  -> jepa_only forward/backward is graph-connected
  * Bug5 (NSA non-causal)   -> perturbation test: future tokens don't leak
  * "test with loss"        -> overfit one batch, total loss decreases
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch

from model.model import AriaModel
from model.nsa import NSAAttention

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def _tiny_model(**over):
    kw = dict(
        d_model=64, n_heads=4, n_loops=4, rank=8,
        nsa=True, nsa_every=3, nsa_block_size=4, nsa_sel_top_n=2,
        window_size=16, sct_kernel=True, sct_fp8=True,
        bitnet_v2=True, bitnet_act_bits=4, bitnet_hadamard=True,
        adaptive_loops=True, jepa=True, mtp=True, mtp_k=4,
    )
    kw.update(over)
    return AriaModel(**kw).to(DEVICE)


def _tiny_batch(B=2, N=4, L=768):
    # patches are byte-level (768 bytes); targets are 16-byte decode patches.
    patches = torch.randint(0, 255, (B, N, L), device=DEVICE).float()
    lengths = torch.randint(1, L, (B, N), device=DEVICE).float()
    is_image = torch.zeros(B, N, dtype=torch.bool, device=DEVICE)
    targets = torch.randint(0, 255, (B, N, 16), device=DEVICE).long()
    return patches, lengths, is_image, targets


def test_joint_ce_jepa_forward_backward():
    model = _tiny_model()
    model.train()
    model.jepa_active = True
    patches, lengths, is_image, targets = _tiny_batch()
    loss = model(patches, lengths, is_image, targets=targets)
    assert torch.isfinite(loss), loss
    loss.backward()
    # grads flow and are finite (Bug1: SCT dx now carries *s)
    n_finite = sum(1 for p in model.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
    assert n_finite > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_jepa_only_forward_backward_graph_connected():
    model = _tiny_model()
    model.train()
    model.jepa_active = True
    model.jepa_only = True
    patches, lengths, is_image, targets = _tiny_batch()
    loss = model(patches, lengths, is_image, targets=targets)
    assert torch.isfinite(loss), loss
    # Bug4: empty-mask path must return a GRAPH-CONNECTED zero, not a detached
    # constant that silently kills the JEPA gradient.
    assert loss.requires_grad
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_nsa_no_future_leak():
    """Perturbing the LAST token must not change earlier outputs (causal)."""
    nsa = NSAAttention(
        d_model=64, n_q_heads=4, n_kv_heads=4, rank=8,
        block_size=4, stride=2, sel_block_size=4, sel_top_n=2, window_size=16,
        bitnet_v2=True, bitnet_act_bits=4, sct_kernel=True,
    ).to(DEVICE).eval()
    T = 16
    x = torch.randn(1, T, 64, device=DEVICE)
    out1 = nsa(x)
    x2 = x.clone()
    x2[0, T - 1] += 10.0  # perturb only the final position
    out2 = nsa(x2)
    # every position before the perturbed one must be bit-identical (no leakage)
    assert torch.allclose(out1[0, :T - 1], out2[0, :T - 1], atol=1e-4)


def test_loss_decreases_on_one_batch():
    model = _tiny_model()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    patches, lengths, is_image, targets = _tiny_batch(B=2, N=6)
    losses = []
    for _ in range(40):
        opt.zero_grad()
        loss = model(patches, lengths, is_image, targets=targets)
        assert torch.isfinite(loss), loss
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.99, (losses[0], losses[-1])


if __name__ == "__main__":
    test_joint_ce_jepa_forward_backward()
    test_jepa_only_forward_backward_graph_connected()
    test_nsa_no_future_leak()
    test_loss_decreases_on_one_batch()
    print("Integration tests: SUCCESS")
