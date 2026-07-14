import torch, torch.nn as nn, torch.nn.functional as F
from .sct import SCTLinear

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=8192):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, seq_len):
        cos = self.cos_cached[:seq_len].view(1, 1, seq_len, -1).to(q.dtype)
        sin = self.sin_cached[:seq_len].view(1, 1, seq_len, -1).to(q.dtype)
        q_rope = (q * cos) + (self._rotate_half(q) * sin)
        k_rope = (k * cos) + (self._rotate_half(k) * sin)
        return q_rope, k_rope

class NSAAttention(nn.Module):
    def __init__(self, d_model, n_q_heads, n_kv_heads, rank=32,
                 block_size=32, stride=16, sel_block_size=32, sel_top_n=16, window_size=512,
                 max_sigma=1.0, fa4=False, sct_kernel=False, sct_fp8=False,
                 bitnet_v2=False, bitnet_act_bits=8, bitnet_hadamard=True):
        bitnet = dict(bitnet_v2=bitnet_v2, bitnet_act_bits=bitnet_act_bits,
                      bitnet_hadamard=bitnet_hadamard)
        super().__init__()
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_model // n_q_heads
        self.block_size = block_size
        self.stride = stride
        self.sel_block_size = sel_block_size
        self.sel_top_n = sel_top_n
        self.window_size = window_size
        self.fa4 = fa4
        dk = d_model * n_kv_heads // n_q_heads
        self.q_proj = SCTLinear(d_model, d_model, rank, max_sigma=max_sigma,
                                 sct_kernel=sct_kernel, sct_fp8=sct_fp8, **bitnet)
        self.k_proj = SCTLinear(d_model, dk, rank, max_sigma=max_sigma,
                                 sct_kernel=sct_kernel, sct_fp8=sct_fp8, **bitnet)
        self.v_proj = SCTLinear(d_model, dk, rank, max_sigma=max_sigma,
                                 sct_kernel=sct_kernel, sct_fp8=sct_fp8, **bitnet)
        self.o_proj = SCTLinear(d_model, d_model, rank, max_sigma=max_sigma,
                                 sct_kernel=sct_kernel, sct_fp8=sct_fp8, **bitnet)
        self.cmp_k_mlp = SCTLinear(d_model // n_q_heads, d_model // n_q_heads, rank, max_sigma=max_sigma,
                                   sct_kernel=sct_kernel, sct_fp8=sct_fp8, **bitnet)
        self.cmp_v_mlp = SCTLinear(d_model // n_q_heads, d_model // n_q_heads, rank, max_sigma=max_sigma,
                                   sct_kernel=sct_kernel, sct_fp8=sct_fp8, **bitnet)
        self.cmp_norm_k = nn.RMSNorm(d_model // n_q_heads)
        self.cmp_norm_v = nn.RMSNorm(d_model // n_q_heads)
        self.gate_mlp = SCTLinear(d_model, d_model, rank, max_sigma=max_sigma,
                                  sct_kernel=sct_kernel, sct_fp8=sct_fp8, **bitnet)
        self.rope = RotaryEmbedding(d_model // n_q_heads)

    def _attn(self, q, k, v, attn_mask=None):
        # q,k,v are (B, H, S, dh). flash-attn-4 wants (B, S, H, dh) and returns a tuple.
        # FA4 is parity (not faster) vs SDPA's flash backend on Blackwell, but the
        # flag is wired for real where the wheel exists (cp314t); SDPA otherwise.
        # A custom (blockwise-causal) mask can't be expressed by FA4's plain-causal
        # flag, so we fall back to SDPA whenever a mask is supplied.
        if self.fa4 and attn_mask is None:
            try:
                from flash_attn.cute import flash_attn_func
                ql, kl, vl = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                return flash_attn_func(ql, kl, vl, causal=True)[0].transpose(1, 2)
            except Exception:
                pass
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

    def compress(self, states, mlp, norm):
        B, H, T, dh = states.shape
        bs = min(self.block_size, T)
        pad_len = (bs - (T % bs)) % bs
        if pad_len > 0:
            states = F.pad(states, (0, 0, 0, pad_len))
        n_blocks = states.shape[2] // bs
        states_blocked = states.view(B, H, n_blocks, bs, dh)
        block_means = states_blocked.mean(dim=3)
        compressed = mlp(block_means.view(-1, dh)).view(B, H, n_blocks, dh)
        return norm(compressed)

    def forward(self, x):
        B, T, D = x.shape
        dh = self.d_head
        hq = self.n_q_heads
        hkv = self.n_kv_heads
        q = self.q_proj(x).view(B, T, hq, dh).transpose(1, 2)
        k = self.k_proj(x).view(B, T, hkv, dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, hkv, dh).transpose(1, 2)
        q, k = self.rope(q, k, T)

        ck = self.compress(k, self.cmp_k_mlp, self.cmp_norm_k)
        cv = self.compress(v, self.cmp_v_mlp, self.cmp_norm_v)
        n_cmp = ck.shape[2]
        ck_exp = ck.unsqueeze(2).expand(B, hkv, hq // max(hkv, 1), n_cmp, dh).reshape(B, hq, n_cmp, dh)
        cv_exp = cv.unsqueeze(2).expand(B, hkv, hq // max(hkv, 1), n_cmp, dh).reshape(B, hq, n_cmp, dh)
        qf = q.reshape(B, hq, T, dh)

        # Blockwise-causal masks: a query at position i may attend a selected/
        # compressed key only if that key's original position <= i (autoregressive;
        # no future leakage). SDPA treats a True mask entry as *masked-out*, so we
        # mark the future as True and use a float -inf additive mask (so a query
        # with no valid past key — e.g. position 0 — gets a zero, not NaN).
        INF = float("-inf")
        qpos = torch.arange(T, device=x.device)
        bs_cmp = min(self.block_size, T)
        cmp_end = torch.min(torch.arange(1, n_cmp + 1, device=x.device) * bs_cmp - 1,
                            torch.tensor(T - 1, device=x.device, dtype=torch.long))
        cmp_keep = qpos.unsqueeze(1) >= cmp_end.unsqueeze(0)   # True where attended
        cmp_mask = torch.where(cmp_keep, torch.zeros((), device=x.device, dtype=torch.float32),
                               torch.full((), INF, device=x.device, dtype=torch.float32))
        o_cmp = self._attn(qf, ck_exp, cv_exp, attn_mask=cmp_mask).transpose(1, 2).reshape(B, T, D)

        top_n = min(self.sel_top_n, n_cmp * 2)
        sel_bs = self.sel_block_size
        sel_k, sel_v, sel_pos = [], [], []
        sink = min(2, n_cmp)
        for i in range(sink):
            start = min(i * self.block_size, max(0, T - sel_bs))
            length = min(sel_bs, T - start)
            if length > 0:
                sel_k.append(k[:, :, start:start + length])
                sel_v.append(v[:, :, start:start + length])
                sel_pos.append(torch.arange(start, start + length, device=x.device))
        step = max(1, min((n_cmp - sink), top_n - sink, 1))
        for i in range(sink, n_cmp, step):
            if len(sel_k) >= top_n: break
            start = min(i * self.block_size, max(0, T - sel_bs))
            length = min(sel_bs, T - start)
            if length > 0:
                sel_k.append(k[:, :, start:start + length])
                sel_v.append(v[:, :, start:start + length])
                sel_pos.append(torch.arange(start, start + length, device=x.device))
        local_start = max(0, T - self.window_size)
        for i in range(max(1, (T - local_start + sel_bs - 1) // sel_bs)):
            if len(sel_k) >= top_n: break
            start = local_start + i * sel_bs
            length = min(sel_bs, T - start)
            if length > 0:
                sel_k.append(k[:, :, start:start + length])
                sel_v.append(v[:, :, start:start + length])
                sel_pos.append(torch.arange(start, start + length, device=x.device))
        if not sel_k:
            sel_k.append(k[:, :, :1])
            sel_v.append(v[:, :, :1])
            sel_pos.append(torch.zeros(1, dtype=torch.long, device=x.device))
        sk = torch.cat(sel_k, dim=2)
        sv = torch.cat(sel_v, dim=2)
        g = hq // max(hkv, 1)
        sk = sk.view(B, hkv, 1, -1, dh).expand(B, hkv, g, -1, dh).reshape(B, hq, -1, dh)
        sv = sv.view(B, hkv, 1, -1, dh).expand(B, hkv, g, -1, dh).reshape(B, hq, -1, dh)

        sel_pos_flat = torch.cat(sel_pos)
        sel_keep = qpos.unsqueeze(1) >= sel_pos_flat.unsqueeze(0)   # True where attended
        sel_mask = torch.where(sel_keep, torch.zeros((), device=x.device, dtype=torch.float32),
                               torch.full((), INF, device=x.device, dtype=torch.float32))
        o_sel = self._attn(qf, sk, sv, attn_mask=sel_mask).transpose(1, 2).reshape(B, T, D)

        # Per-position gate: gf must depend ONLY on this position's own
        # o_cmp/o_sel. A sequence mean here (the original code) injects future
        # context into every position's gate and breaks causality.
        gf = o_cmp + o_sel
        gates = torch.sigmoid(self.gate_mlp(gf))
        out = o_cmp * gates + o_sel * (1 - gates)
        return self.o_proj(out)
