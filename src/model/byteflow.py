import torch
import torch.nn as nn
import torch.nn.functional as F
from .sct import SCTLinear

# Extended byte vocab: 0-255 raw bytes, 256-267 special tokens, 268 <pad>.
VOCAB_SIZE = 269
PAD_ID = 268


class ByteFlowEncoder(nn.Module):
    """ByteFlow: tokenizer-free early-fusion byte encoder.

    Unifies text (dynamic patches) and images (flat 768-byte blocks) into one
    d_model latent stream. Text bytes use the extended vocab (0-268); images
    are raw 16x16x3 pixels.
    """

    def __init__(self, d_model=768, d_byte=128, max_patch_len=16, rank=32, max_sigma=1.0):
        super().__init__()
        self.d_model = d_model
        self.d_byte = d_byte
        self.max_patch_len = max_patch_len
        self.byte_embed = nn.Embedding(VOCAB_SIZE, d_byte, padding_idx=PAD_ID)
        self.local_conv = nn.Conv1d(d_byte, d_byte, kernel_size=3, padding=1)
        self.boundary_head = nn.Linear(d_byte, 1)
        self.compressor = SCTLinear(d_byte, d_model, rank=rank, max_sigma=max_sigma)
        self.image_proj = SCTLinear(768, d_model, rank=rank, max_sigma=max_sigma)
        self.modality_indicator = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, patches, is_image_mask):
        B, T, L = patches.shape
        dtype = self.image_proj.U.dtype

        text_bytes = patches[:, :, : self.max_patch_len].long()
        embedded = self.byte_embed(text_bytes).to(dtype)
        flat_embed = embedded.view(-1, self.max_patch_len, self.d_byte).transpose(1, 2)
        conv_feats = F.silu(self.local_conv(flat_embed)).transpose(1, 2)

        boundary_scores = torch.sigmoid(self.boundary_head(conv_feats)).squeeze(-1)
        weights = F.softmax(boundary_scores, dim=-1).unsqueeze(-1)
        pooled_text = torch.sum(conv_feats * weights, dim=1).view(B, T, self.d_byte)
        text_latents = self.compressor(pooled_text)

        image_pixels = patches.to(dtype) / 255.0
        image_latents = self.image_proj(image_pixels)

        out = torch.where(is_image_mask.unsqueeze(-1), image_latents, text_latents)
        out = out + torch.where(is_image_mask.unsqueeze(-1), self.modality_indicator, torch.zeros_like(out))
        return out


class ByteFlowDecoder(nn.Module):
    """Decode HelixCore latents back to extended bytes (0-268) or raw RGB pixels."""

    def __init__(self, d_model=768, rank=32, max_sigma=1.0, max_patch_len=16):
        super().__init__()
        self.d_model = d_model
        self.max_patch_len = max_patch_len
        self.decoder_head = SCTLinear(d_model, 128, rank=rank, max_sigma=max_sigma)
        self.image_head = nn.Linear(128, 768, bias=False)
        # ponytail: GRU cell as plain linear gates (no nn.GRUCell — its
        # _thnn_fused_gru_cell autograd op breaks torch.compile meta-tensors).
        self.gru_ih = nn.Linear(128, 384, bias=False)
        self.gru_hh = nn.Linear(128, 384, bias=False)
        self.classifier = nn.Linear(128, VOCAB_SIZE)

    def forward(self, x, is_image_mask, target_bytes=None):
        B, N, D = x.shape
        L = self.max_patch_len
        flat_x = x.view(-1, D)
        init_states = F.silu(self.decoder_head(flat_x))

        # ponytail: dropped the `if bool(is_image_mask.all())` image-pixel
        # early-return — its device→host sync broke CUDA graph capture, and no
        # caller consumes decoder image pixels for our text data (run_encoded
        # discards them). image_head is kept so old checkpoints still load.
        # Re-add as a host-computed static flag if image-only batches appear.
        logits_list = []
        current_state = init_states
        input_emb = torch.zeros_like(init_states)
        for _ in range(L):
            gates = self.gru_ih(input_emb) + self.gru_hh(current_state)
            r, z, n = gates.chunk(3, dim=-1)
            r, z = torch.sigmoid(r), torch.sigmoid(z)
            n = torch.tanh(n)
            current_state = (1 - z) * n + z * current_state
            step_logits = self.classifier(current_state)
            logits_list.append(step_logits)
            # ponytail: closed-loop rollout; teacher forcing left unwired (dump's
            # version fed current_state anyway, so it was a no-op). Wire later if
            # training stability needs it.
            input_emb = current_state
        all_logits = torch.stack(logits_list, dim=2).view(B, N, L, VOCAB_SIZE)
        return all_logits, None
