"""Aria-JEPA: I-JEPA masked latent prediction + Semantic Tube Prediction (STP).

Design (quality bar, full core integration — not a patch):
  * Context encoder = the SAME AriaModel trunk (ByteFlowEncoder + HelixCore).
    For a masked-context view we inject a learned [MASK] embedding at masked
    patch positions, then run the trunk normally (shared weights, stop-grad on
    the target view). No separate encoder, no wrapper.
  * Two specialized predictors, one blended by a learned router:
      - KnowledgePredictor (SwiGLU / SCTLinear): associative / factual.
      - LogicPredictor    (HFW-KANLayer)        : structured logic / math.
  * Loss = MSE(pred, target_rep) for each predictor  +  VICReg on the blended
    prediction (variance + covariance regularizers kill latent collapse) +
    STP curvature penalty on the HelixCore loop trajectory.
  * Target reps come from a no-grad full forward; only the online (masked) view
    and its predicted trajectory carry gradient, so the encoder/helix learn a
    world model by predicting the masked future from context.

References:
  I-JEPA        — Assran et al., arXiv:2301.08243
  Semantic Tube — arXiv:2602.22617 (LANG-JEPA, May 2026)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sct import SCTLinear
from .hwf_kan import HFW_KANLayer


def variance_loss(x, gamma=1.0, eps=1e-4):
    """VICReg variance term: push per-dim std above `gamma` (anti-collapse). x:[N,D]."""
    std = torch.sqrt(x.var(dim=0) + eps)
    return torch.mean(F.relu(gamma - std))


def covariance_loss(x):
    """VICReg covariance term: decorrelate dims (anti-redundancy). x:[N,D]."""
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / (x.shape[0] - 1)
    off = cov - torch.diag(torch.diag(cov))
    return (off ** 2).sum() / x.shape[1]


class KnowledgePredictor(nn.Module):
    """Associative / factual predictor — spectrally-stable SwiGLU on SCTLinear."""

    def __init__(self, d_model, hidden, rank, max_sigma, sct_kernel, sct_fp8,
                 bitnet_v2, bitnet_act_bits, bitnet_hadamard):
        super().__init__()
        self.w1 = SCTLinear(d_model, hidden, rank=rank, max_sigma=max_sigma,
                            sct_kernel=sct_kernel, sct_fp8=sct_fp8,
                            bitnet_v2=bitnet_v2, bitnet_act_bits=bitnet_act_bits,
                            bitnet_hadamard=bitnet_hadamard)
        self.w2 = SCTLinear(d_model, hidden, rank=rank, max_sigma=max_sigma,
                            sct_kernel=sct_kernel, sct_fp8=sct_fp8,
                            bitnet_v2=bitnet_v2, bitnet_act_bits=bitnet_act_bits,
                            bitnet_hadamard=bitnet_hadamard)
        self.w3 = SCTLinear(hidden, d_model, rank=rank, max_sigma=max_sigma,
                            sct_kernel=sct_kernel, sct_fp8=sct_fp8,
                            bitnet_v2=bitnet_v2, bitnet_act_bits=bitnet_act_bits,
                            bitnet_hadamard=bitnet_hadamard)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class LogicPredictor(nn.Module):
    """Structured logic / math predictor — hyperbolic wavelet-filtered KAN."""

    def __init__(self, d_model, degree, num_frequencies, max_sigma, fp8_kan):
        super().__init__()
        self.kan = HFW_KANLayer(d_model, d_model, degree, num_frequencies,
                                max_sigma=max_sigma, fp8=fp8_kan)

    def forward(self, x):
        return self.kan(x)


class JEPARouter(nn.Module):
    """Per-position scalar gate in [0,1] blending Knowledge/Logic predictors."""

    def __init__(self, d_model, rank, max_sigma, sct_kernel, sct_fp8,
                 bitnet_v2, bitnet_act_bits, bitnet_hadamard):
        super().__init__()
        self.pre = SCTLinear(d_model, d_model, rank=rank, max_sigma=max_sigma,
                             sct_kernel=sct_kernel, sct_fp8=sct_fp8,
                             bitnet_v2=bitnet_v2, bitnet_act_bits=bitnet_act_bits,
                             bitnet_hadamard=bitnet_hadamard)
        self.gate = SCTLinear(d_model, 1, rank=rank, max_sigma=max_sigma,
                              sct_kernel=sct_kernel, sct_fp8=sct_fp8,
                              bitnet_v2=bitnet_v2, bitnet_act_bits=bitnet_act_bits,
                              bitnet_hadamard=bitnet_hadamard)

    def forward(self, x):
        return torch.sigmoid(self.gate(F.silu(self.pre(x))))


class AriaJEPA(nn.Module):
    def __init__(self, d_model, pred_hidden=1024, rank=32, max_sigma=1.0,
                 sct_kernel=False, sct_fp8=False, bitnet_v2=False,
                 bitnet_act_bits=8, bitnet_hadamard=True, fp8_kan=False,
                 patch_size=8, context_keep=0.7, degree=6, num_frequencies=3):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.context_keep = context_keep
        self.knowledge = KnowledgePredictor(d_model, pred_hidden, rank, max_sigma,
                                            sct_kernel, sct_fp8, bitnet_v2,
                                            bitnet_act_bits, bitnet_hadamard)
        self.logic = LogicPredictor(d_model, degree, num_frequencies, max_sigma, fp8_kan)
        self.router = JEPARouter(d_model, rank, max_sigma, sct_kernel, sct_fp8,
                                 bitnet_v2, bitnet_act_bits, bitnet_hadamard)

    def forward(self, x):
        """x: [*, N, D] -> (pred_k, pred_l, gate, rep_on)."""
        pred_k = self.knowledge(x)
        pred_l = self.logic(x)
        gate = self.router(x)
        rep_on = gate * pred_k + (1 - gate) * pred_l
        return pred_k, pred_l, gate, rep_on

    def build_context_mask(self, patches, is_image_mask, lengths=None):
        """Block-wise masking over TEXT positions only.

        Returns (target_mask, keep_mask, n_blocks, T). target_mask: [B,T] bool,
        True at masked positions. Image / padding positions are never masked.
        """
        B, T, _ = patches.shape
        device = patches.device
        C = max(1, T // self.patch_size)
        target_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        if C >= 2:
            num_mask = min(C - 1, int(round(C * (1 - self.context_keep))))
            eligible = ~is_image_mask
            if lengths is not None:
                pos = torch.arange(T, device=device).unsqueeze(0)
                eligible = eligible & (pos < lengths.long().unsqueeze(1))
            block_starts = torch.arange(0, T, self.patch_size, device=device)
            for b in range(B):
                cand = []
                for i, s in enumerate(block_starts):
                    e = min(s + self.patch_size, T)
                    if eligible[b, s:e].all():
                        cand.append(i)
                if not cand:
                    continue
                perm = torch.randperm(len(cand), device=device)[:num_mask]
                for p in perm.tolist():
                    s = int(block_starts[cand[p]].item())
                    e = min(s + self.patch_size, T)
                    target_mask[b, s:e] = True
        return target_mask, ~target_mask, C, T

    def jepa_loss(self, pred_k, pred_l, rep_on, target_rep):
        k = F.mse_loss(pred_k, target_rep)
        l = F.mse_loss(pred_l, target_rep)
        r = rep_on.reshape(-1, rep_on.shape[-1])
        v = variance_loss(r)
        c = covariance_loss(r)
        return k, l, v, c

    def semantic_tube(self, traj):
        """STP curvature penalty: mean squared 2nd-difference over loop steps.

        traj: list of [B,T,D] per HelixCore loop step (len = n_loops).
        Returns 0 for a perfectly linear (constant-velocity) trajectory.
        """
        if len(traj) < 3:
            return torch.zeros((), device=traj[0].device, dtype=traj[0].dtype)
        x = torch.stack(traj, dim=0)            # [L,B,T,D]
        acc = x[2:] - 2 * x[1:-1] + x[:-2]     # [L-2,B,T,D]
        return (acc ** 2).mean()
