import torch
import torch.nn as nn
import torch.nn.functional as F
from .byteflow import ByteFlowEncoder, ByteFlowDecoder
from .helix import HelixCore
from .nsa import NSAAttention

VOCAB_SIZE = 269
PAD_ID = 268


class AriaModel(nn.Module):
    def __init__(self, d_model=768, n_heads=12, n_loops=6, rank=32,
                 nsa=False, window_size=512, nsa_block_size=32, nsa_sel_top_n=16,
                 use_cld=False, cld_gamma=0.1, gradient_checkpoint_every=4,
                 compile=False, degree=6, num_frequencies=3, temperature=1.0,
                 ponder_lambda=0.01, max_sigma=1.0, sct_kernel=False, sct_fp8=False,
                 fa4=False, dropbp=0.0, lcsb_ratio=0.0, fp8_kan=False, use_checkpointing=True,
                 bitnet_v2=False, bitnet_act_bits=8, bitnet_hadamard=True,
                 loop_checkpoint=False, mixture_of_depths=False, mod_capacity=0.5,
                 adaptive_loops=False, engram_vocab_size=65536):
        super().__init__()
        self.d_model = d_model
        self.use_cld = use_cld
        self.cld_gamma = cld_gamma
        self.ponder_lambda = ponder_lambda
        self.engram_vocab_size = engram_vocab_size
        # Tokenizer-free: no embed_w, no AnchorBlock. Bytes enter via ByteFlowEncoder.
        self.encoder = ByteFlowEncoder(d_model, rank=rank, max_sigma=max_sigma)
        self.decoder = ByteFlowDecoder(d_model, rank=rank, max_sigma=max_sigma, max_patch_len=16)
        self.helix = HelixCore(d_model, max_loops=n_loops, degree=degree,
                               num_frequencies=num_frequencies, temperature=temperature,
                               max_sigma=max_sigma, dropbp=dropbp, lcsb_ratio=lcsb_ratio,
                                sct_kernel=sct_kernel, sct_fp8=sct_fp8, fp8_kan=fp8_kan,
                                use_checkpointing=use_checkpointing,
                                bitnet_v2=bitnet_v2, bitnet_act_bits=bitnet_act_bits,
                                bitnet_hadamard=bitnet_hadamard,
                                loop_checkpoint=loop_checkpoint,
                                mixture_of_depths=mixture_of_depths,
                                mod_capacity=mod_capacity,
                                adaptive_loops=adaptive_loops,
                                engram_vocab_size=engram_vocab_size)
        if compile:
            mode = 'max-autotune' if compile is True else compile
            self.helix = torch.compile(self.helix, mode=mode)
        if nsa:
            self.use_nsa = True
            self.cross_attn = NSAAttention(d_model, n_heads, n_heads, rank,
                                            nsa_block_size, nsa_block_size // 2,
                                            nsa_block_size, nsa_sel_top_n, window_size,
                                            max_sigma=max_sigma, fa4=fa4,
                                            sct_kernel=sct_kernel, sct_fp8=sct_fp8,
                                            bitnet_v2=bitnet_v2, bitnet_act_bits=bitnet_act_bits,
                                            bitnet_hadamard=bitnet_hadamard)
        else:
            self.use_nsa = False

    def forward(self, patches, patch_lengths, is_image_mask, targets=None, active_loops=None, cld_gamma=None, return_halt=False):
        x = self.encoder(patches, is_image_mask)
        return self.run_encoded(x, patches, is_image_mask, targets, active_loops, cld_gamma, return_halt)

    def run_encoded(self, x, patches, is_image_mask, targets=None, active_loops=None, cld_gamma=None, return_halt=False):
        flat_tokens = patches.long()[:, :, 0]
        return_shallow = self.use_cld and self.training and (
            cld_gamma if cld_gamma is not None else self.cld_gamma) > 0
        shallow_step = min(active_loops or 6, 2) if return_shallow else None
        h, halt_probs, h_shallow = self.helix(x, flat_tokens, max_loops=active_loops,
                                              return_shallow_at=shallow_step)
        if self.use_nsa:
            h = x + self.cross_attn(h)
            if return_shallow and h_shallow is not None:
                h_shallow = x + self.cross_attn(h_shallow)
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
            # Fall back to ponder loss in that case.
            loss = ce if torch.isfinite(ce) else torch.zeros_like(ce)
            if self.training and len(halt_probs) > 0:
                remaining = 1.0 - torch.stack(halt_probs, dim=0).sum(dim=0)
                loss = loss + self.ponder_lambda * remaining.mean()
            return (loss, halt_probs) if return_halt else loss
        logits, pixels = self.decoder(h, is_image_mask)
        return (logits, pixels, halt_probs) if return_halt else (logits, pixels)
