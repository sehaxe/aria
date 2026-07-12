import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSeekEngram(nn.Module):
    """
    Conditional memory (DeepSeek Engram): O(1) static n-gram recall injected
    into the recurrent core so HelixCore can spend capacity on reasoning instead
    of orthographic routine.
    """

    def __init__(self, d_model, engram_vocab_size=65536, max_ngram=3, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.engram_vocab_size = engram_vocab_size
        self.max_ngram = max_ngram
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.embedding_table = nn.Embedding(engram_vocab_size, self.head_dim)
        # Default nn.Embedding init is uninitialized / hot for giant vocab;
        # scale to LLaMA/GPT standard (std=0.02) to avoid bf16 overflow.
        nn.init.normal_(self.embedding_table.weight, mean=0.0, std=0.02)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.norm = nn.RMSNorm(self.head_dim)
        # depthwise conv over the gated memory; padded on the LEFT only => causal
        self.short_conv = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)

    def _ngram_hashes(self, tokens):
        B, T = tokens.shape
        compressed = tokens % 128
        hashes = [compressed % self.engram_vocab_size]
        if T > 1:
            h2 = (compressed[:, :-1] * 31 + compressed[:, 1:]) % self.engram_vocab_size
            hashes.append(F.pad(h2, (1, 0), value=0))
        if T > 2:
            h3 = (compressed[:, :-2] * 961 + compressed[:, 1:-1] * 31 + compressed[:, 2:]) % self.engram_vocab_size
            hashes.append(F.pad(h3, (2, 0), value=0))
        stacked = torch.stack(hashes[: self.num_heads], dim=-1)
        if stacked.shape[-1] < self.num_heads:
            stacked = F.pad(stacked, (0, self.num_heads - stacked.shape[-1]), value=0)
        return stacked

    def forward(self, hidden_states, input_ids):
        B, T, D = hidden_states.shape
        dtype = hidden_states.dtype
        hashes = self._ngram_hashes(input_ids)
        # embedding lookup stays fp32; cast to the active dtype for the rest
        embeddings = self.embedding_table(hashes).to(dtype)
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(self.norm(embeddings)).view(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(embeddings).view(B, T, self.num_heads, self.head_dim)
        # context-aware gate: hidden state acts as query over the recalled memory
        gate = torch.sigmoid(torch.sum(q * k, dim=-1, keepdim=True))
        gated = (gate * v).reshape(B, T, D)
        # causal local smoothing (left-pad so position t only sees t-k..t)
        conv_in = gated.transpose(1, 2)
        conv_out = self.short_conv(F.pad(conv_in, (self.short_conv.kernel_size[0] - 1, 0)))[..., :T]
        conv_out = conv_out.transpose(1, 2)
        return (gated + conv_out).to(dtype)
