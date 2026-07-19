import torch
import torch.nn as nn
import torch.nn.functional as F
from .byteflow import ByteFlowEncoder, ByteFlowDecoder
from .helix import HelixCore
from .nsa import NSAAttention
from .jepa import AriaJEPA
from .sct import SCTLinear

VOCAB_SIZE = 269
PAD_ID = 268


class AriaModel(nn.Module):
    def __init__(self, d_model=768, n_heads=12, n_loops=6, rank=32,
                 nsa=False, nsa_every=3, window_size=512, nsa_block_size=32, nsa_sel_top_n=16,
                 gradient_checkpoint_every=4,
                 compile=False, degree=6, num_frequencies=3, temperature=1.0,
                 ponder_lambda=0.01, max_sigma=1.0, sct_kernel=False, sct_fp8=False,
                 fa4=False, dropbp=0.0, lcsb_ratio=0.0, fp8_kan=False, use_checkpointing=True,
                 bitnet_v2=False, bitnet_act_bits=8, bitnet_hadamard=True,
                 loop_checkpoint=False, mixture_of_depths=False, mod_capacity=0.5,
                  adaptive_loops=False, engram_vocab_size=65536,
                 jepa=False, jepa_pred_hidden=1024, jepa_context_keep=0.7,
                 jepa_patch_size=8, jepa_lambda_k=1.0, jepa_lambda_l=1.0,
                 jepa_stp=0.1, jepa_vicreg_var=1.0, jepa_vicreg_cov=1.0,
                  jepa_dropout=0.5, jepa_kl_coef=0.01,
                  speculative=True, speculative_k=4, speculative_loss_coef=0.1,
                  mtp=True, mtp_k=4, mtp_loss_coef=0.1,
                  worldmodel_halt=True, forecaster_loss_coef=0.1, **_):
        super().__init__()
        self.d_model = d_model
        self.ponder_lambda = ponder_lambda
        self.engram_vocab_size = engram_vocab_size
        # Tokenizer-free: no embed_w, no AnchorBlock. Bytes enter via ByteFlowEncoder.
        self.encoder = ByteFlowEncoder(d_model, rank=rank, max_sigma=max_sigma)
        self.decoder = ByteFlowDecoder(d_model, rank=rank, max_sigma=max_sigma, max_patch_len=16)
        self.use_nsa = nsa
        if nsa:
            self.cross_attn = NSAAttention(d_model, n_heads, n_heads, rank,
                                            nsa_block_size, nsa_block_size // 2,
                                            nsa_block_size, nsa_sel_top_n, window_size,
                                            max_sigma=max_sigma, fa4=fa4,
                                            sct_kernel=sct_kernel, sct_fp8=sct_fp8,
                                            bitnet_v2=bitnet_v2, bitnet_act_bits=bitnet_act_bits,
                                            bitnet_hadamard=bitnet_hadamard)
        else:
            self.cross_attn = None
        self.helix = HelixCore(d_model, max_loops=n_loops, degree=degree,
                               num_frequencies=num_frequencies, temperature=temperature,
                               max_sigma=max_sigma, dropbp=dropbp, lcsb_ratio=lcsb_ratio,
                                sct_kernel=sct_kernel, sct_fp8=sct_fp8, fp8_kan=fp8_kan,
                                use_checkpointing=use_checkpointing,
                                bitnet_v2=bitnet_v2, bitnet_act_bits=bitnet_act_bits,
                                bitnet_hadamard=bitnet_hadamard,
                                loop_checkpoint=use_checkpointing,
                                mixture_of_depths=mixture_of_depths,
                                 mod_capacity=mod_capacity,
                                 adaptive_loops=adaptive_loops,
                                  worldmodel_halt=worldmodel_halt,
                                  forecaster_loss_coef=forecaster_loss_coef,
                                  engram_vocab_size=engram_vocab_size,
                                  nsa_attn=(self.cross_attn if nsa else None), nsa_every=nsa_every)
        if compile:
            mode = 'max-autotune' if compile is True else compile
            self.helix = torch.compile(self.helix, mode=mode)
        # A: self-speculative decoding draft head. Trained (teacher-forced) to
        # forecast the next patch from (hidden, prev-patch); at inference it
        # drafts K patches cheaply and the trunk verifies them in one forward.
        self.speculative = speculative
        self.speculative_k = speculative_k
        self.speculative_loss_coef = speculative_loss_coef
        self.mtp = mtp
        self.mtp_k = mtp_k
        self.mtp_loss_coef = mtp_loss_coef
        # MTP-4 (DeepSeek-V3 style): k heads predict future patches t+1..t+mtp_k
        # from the trunk hidden; head 0 == the self-speculative draft head (A) so
        # inference drafting and MTP-1 share weights. Registered once: head 0 under
        # `draft_hidden`, offsets 2..k under `mtp_heads` (no shared-instance double
        # registration).
        self.draft_hidden = SCTLinear(d_model, d_model, rank=rank, max_sigma=max_sigma,
                                      sct_kernel=sct_kernel, sct_fp8=sct_fp8,
                                      bitnet_v2=bitnet_v2, bitnet_act_bits=bitnet_act_bits,
                                      bitnet_hadamard=bitnet_hadamard) if (speculative or mtp) else None
        if mtp and mtp_k > 1:
            self.mtp_heads = nn.ModuleList([
                SCTLinear(d_model, d_model, rank=rank, max_sigma=max_sigma,
                          sct_kernel=sct_kernel, sct_fp8=sct_fp8,
                          bitnet_v2=bitnet_v2, bitnet_act_bits=bitnet_act_bits,
                          bitnet_hadamard=bitnet_hadamard)
                for _ in range(mtp_k - 1)])
        else:
            self.mtp_heads = None
        self.use_nsa = nsa

        # Aria-JEPA: I-JEPA masked latent prediction + Semantic Tube Prediction.
        # Integrated into the trunk (dual forward, shared weights) — not a wrapper.
        self.jepa = AriaJEPA(d_model, jepa_pred_hidden, rank, max_sigma, sct_kernel,
                             sct_fp8, bitnet_v2, bitnet_act_bits, bitnet_hadamard,
                             fp8_kan, jepa_patch_size, jepa_context_keep, degree,
                             num_frequencies) if jepa else None
        if self.jepa is not None:
            self.mask_emb = nn.Parameter(torch.zeros(d_model))
        self.jepa_active = False   # toggled per training stage (train_phased)
        self.jepa_only = False     # stage trains only the world model (no CE)
        self.last_jepa_aux = {}
        self.jepa_lambda_k = jepa_lambda_k
        self.jepa_lambda_l = jepa_lambda_l
        self.jepa_vicreg_var = jepa_vicreg_var
        self.jepa_vicreg_cov = jepa_vicreg_cov
        self.jepa_stp = jepa_stp
        self.jepa_kl_coef = jepa_kl_coef
        self.jepa_dropout = jepa_dropout

    def forward(self, patches, patch_lengths, is_image_mask, targets=None, active_loops=None, return_halt=False, return_hidden=False):
        # Drop cached quantized weights from the previous step so each forward
        # builds a fresh autograd graph (reused tensors from a freed graph are
        # unsafe). The per-loop cache in SCTLinear.forward still collapses the
        # 48 loop re-quantizations down to one per step.
        for m in self.modules():
            if isinstance(m, SCTLinear):
                m._uq = m._vq = None
        x = self.encoder(patches, is_image_mask)
        jepa_loss = None
        # Aria-JEPA world-modeling branch: dual forward (masked online + full target).
        if self.jepa is not None and self.training and self.jepa_active:
            do_jepa = True if self.jepa_only else (
                torch.rand(1, device=x.device) < self.jepa_dropout).item()
            if do_jepa:
                jepa_loss, aux = self._jepa_step(x, patches, is_image_mask,
                                                patch_lengths, active_loops)
                self.last_jepa_aux = aux or {}
                if self.jepa_only:
                    return jepa_loss
        out = self.run_encoded(x, patches, is_image_mask, targets, active_loops, return_halt, return_hidden=return_hidden)
        if self.training and targets is not None and jepa_loss is not None and not return_hidden:
            if return_halt:
                loss, hp = out
                return (loss + jepa_loss, hp)
            return out + jepa_loss
        return out

    def _jepa_step(self, x, patches, is_image_mask, patch_lengths, active_loops):
        """I-JEPA dual forward: predict masked target reps from context; STP on trajectory."""
        target_mask, keep, C, T = self.jepa.build_context_mask(patches, is_image_mask, patch_lengths)
        if target_mask.sum() == 0:
            # No masked positions this batch: return a graph-connected zero so
            # backward (incl. jepa_only) is a harmless no-op instead of a detached
            # constant that silently kills the JEPA gradient.
            return (self.mask_emb * 0).sum(), {}
        # online (masked context) forward — carries gradient + trajectory
        _, _, _, h_online, traj_online = self.run_encoded(x, patches, is_image_mask,
                                                  active_loops=active_loops,
                                                  patch_mask=target_mask,
                                                  return_hidden=True, record_traj=True)
        # target (full) forward — stop-grad reference representations
        with torch.no_grad():
            _, _, _, h_target, _ = self.run_encoded(x, patches, is_image_mask,
                                           active_loops=active_loops, return_hidden=True)
            tg_all = h_target[target_mask].detach()
        on_all = h_online[target_mask]
        pred_k, pred_l, gate, rep_on = self.jepa(on_all.unsqueeze(0))
        k, l, v, c = self.jepa.jepa_loss(pred_k, pred_l, rep_on, tg_all.unsqueeze(0))
        L_jepa = (self.jepa_lambda_k * k + self.jepa_lambda_l * l
                  + self.jepa_vicreg_var * v + self.jepa_vicreg_cov * c)
        L_tube = self.jepa.semantic_tube(traj_online) if self.jepa_stp > 0 else torch.zeros((), device=x.device)
        balance = (gate.mean() - 0.5).pow(2)   # keep both predictors engaged
        total = L_jepa + self.jepa_stp * L_tube + self.jepa_kl_coef * balance
        aux = {"jepa_k": float(k.item()), "jepa_l": float(l.item()),
               "jepa_v": float(v.item()), "jepa_c": float(c.item()),
               "stp": float(L_tube.item()), "gate": float(gate.mean().item())}
        return total, aux

    def run_encoded(self, x, patches, is_image_mask, targets=None, active_loops=None,
                    return_halt=False, patch_mask=None,
                    return_hidden=False, record_traj=False):
        flat_tokens = patches.long()[:, :, 0]
        if patch_mask is not None and self.mask_emb is not None:
            x = x.clone()
            x[patch_mask] = self.mask_emb   # learned [MASK] embedding at masked patches
        compute_forecaster = (targets is not None) and (not self.jepa_only)
        h, halt_probs, state_codes = self.helix(
            x, flat_tokens, max_loops=active_loops,
            record_traj=record_traj, compute_forecaster=compute_forecaster)
        # ponytail: NSA is interleaved inside HelixCore._run_loop (every
        # nsa_every GDN2 steps), so no post-loop attention pass is needed here.
        if targets is not None:
            # Mixed batch: push image patches to pad(268) so CE ignores them.
            # The forward pass still runs on pixels; only the text loss is masked.
            # ponytail: targets are full 768-byte patches; decoder produces 16
            # logits per patch (max_patch_len). Slice to match.
            target_bytes = targets.long()[:, :, :self.decoder.max_patch_len]
            masked_targets = torch.where(is_image_mask.unsqueeze(-1), PAD_ID, target_bytes)
            logits, _ = self.decoder(h, is_image_mask, target_bytes=masked_targets)
            ce = F.cross_entropy(logits.view(-1, VOCAB_SIZE), masked_targets.view(-1),
                                 ignore_index=PAD_ID)
            # ponytail: CE=NaN when ALL targets are PAD (image-only batch).
            # Fall back to ponder loss in that case (branchless for torch.compile).
            loss = torch.where(torch.isfinite(ce), ce, torch.zeros_like(ce))
            if self.training and len(halt_probs) > 0:
                remaining = 1.0 - torch.stack(halt_probs, dim=0).sum(dim=0)
                loss = loss + self.ponder_lambda * remaining.mean()
            if targets is not None and h is not None and h.shape[1] >= 2 and (self.mtp or self.speculative):
                if self.mtp:
                    loss = loss + self._mtp_loss(h, x, patches, is_image_mask)
                elif self.speculative:
                    loss = loss + self.speculative_loss_coef * self._draft_loss(h, x, patches, is_image_mask)
            fl = getattr(self.helix, "last_forecaster_loss", None)
            # ponytail: branchless add (fl is 0 when forecaster inactive); the
            # Python float() guard breaks torch.compile (data-dependent).
            loss = loss + (fl if fl is not None else torch.zeros_like(loss))
            if return_hidden:
                return (loss, halt_probs, h, state_codes)
            return (loss, halt_probs) if return_halt else loss
        logits, pixels = self.decoder(h, is_image_mask)
        if return_hidden:
            return (logits, pixels, halt_probs, h, state_codes)
        return (logits, pixels, halt_probs) if return_halt else (logits, pixels)

    def _draft_loss(self, h, x, patches, is_image_mask):
        """Self-speculative draft aux loss (Medusa-style, teacher-forced).

        From (trunk hidden h_t, real patch embedding x_t) predict patch t+1.
        Trains the draft head to forecast; at inference the same head chains
        autoregressively over DRAFTED patches, which the trunk then verifies.
        """
        if h is None or h.shape[1] < 2:
            return torch.zeros((), device=h.device if h is not None else patches.device,
                               dtype=patches.dtype)
        h_in = h[:, :-1] + x[:, :-1]
        h_next = self.draft_hidden(h_in)
        draft_logits = self.decoder(h_next, is_image_mask[:, :-1])[0]
        tgt = patches[:, 1:, :self.decoder.max_patch_len].long()
        return F.cross_entropy(draft_logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1),
                                ignore_index=PAD_ID)

    def _mtp_loss(self, h, x, patches, is_image_mask):
        """Multi-Token Prediction aux loss (DeepSeek-V3 style).

        From the trunk hidden h_t predict patches t+1..t+mtp_k via k heads; each
        head is teacher-forced on the ground-truth next patch. Weighted 0.5^(k-1)
        so near-term predictions dominate. Training-only (targets present).
        """
        if not self.training or self.draft_hidden is None or h is None or h.shape[1] < 2:
            return torch.zeros((), device=h.device, dtype=h.dtype)
        B, T, D = h.shape
        total = torch.zeros((), device=h.device, dtype=h.dtype)
        for k in range(1, self.mtp_k + 1):
            if T <= k:
                break
            h_in = h[:, :T - k] + x[:, :T - k]
            head = self.draft_hidden if k == 1 else self.mtp_heads[k - 2]
            h_pred = head(h_in)
            logits = self.decoder(h_pred, is_image_mask[:, :T - k])[0]
            tgt = patches[:, k:T, :self.decoder.max_patch_len].long()
            masked = torch.where(is_image_mask[:, k:T].unsqueeze(-1), PAD_ID, tgt)
            ce = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), masked.reshape(-1),
                                 ignore_index=PAD_ID)
            total = total + (0.5 ** (k - 1)) * (ce if torch.isfinite(ce) else torch.zeros_like(ce))
        return total * self.mtp_loss_coef
