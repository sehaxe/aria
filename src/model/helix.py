import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .hwf_kan import DynamicFFN
from .depth_lora import DepthLoRA
from .lti_injection import LTIInjection
from .loop_ln import LoopModulatedLN
from .sct import SCTLinear
from .engram import DeepSeekEngram
from kernels.fused_gru_lti import FusedGRULTI
from mask import combined_mask
from bitnet_v2 import bitnet_int_codes, _dequant, _HadamardTransform


class _BitnetRecStep(torch.autograd.Function):
    """Int-storage recurrent checkpoint (BitNet v2, no torch.checkpoint).

    Stores the recurrent hidden state h and the quantized input x_encoded as int
    activation codes — not bf16 — so the between-step state is tiny. Backward
    recomputes the step from the int codes (STE through the quant, exact through
    the orthogonal Hadamard) instead of holding all loop steps' bf16 activations.
    This is what lets BitNet train without torch checkpointing.
    """

    @staticmethod
    def forward(ctx, helix, h, x_enc, xq, xs, xp, engram_mem, step, prev_conf, mask,
                is_last, bits, hadamard):
        hq, hs, hp = bitnet_int_codes(h, bits, hadamard)
        h_dq = _dequant(hq, hs, bits, hp)
        x_dq = _dequant(xq, xs, bits, xp)
        ctx.helix = helix
        ctx.bits, ctx.hadamard = bits, hadamard
        ctx.hp, ctx.xp = hp, xp
        ctx.step, ctx.mask, ctx.is_last = step, mask, is_last
        # ponytail: constant (non-grad) inputs kept on ctx; only h carries grad.
        # x codes are the shared loop constant passed in (one allocation).
        ctx.engram_mem = engram_mem
        ctx.prev_conf = prev_conf
        ctx.save_for_backward(hq, hs, xq, xs)   # int codes only (tiny)
        h_next, halt = helix._loop_step(h_dq, x_dq, engram_mem, step, prev_conf)
        if mask is not None:
            h_det = h_dq + (h_next - h_dq).detach()
            h_next = torch.where(mask[step], h_det, h_next)
        if is_last:
            halt = torch.ones_like(halt)
        return h_next, halt

    @staticmethod
    def backward(ctx, g_hnext, g_halt):
        helix = ctx.helix
        hq, hs, xq, xs = ctx.saved_tensors
        h_dq = _dequant(hq, hs, ctx.bits, ctx.hp).requires_grad_(True)
        x_dq = _dequant(xq, xs, ctx.bits, ctx.xp).requires_grad_(True)
        with torch.enable_grad():
            h_next, halt = helix._loop_step(h_dq, x_dq, ctx.engram_mem,
                                            ctx.step, ctx.prev_conf)
            if ctx.mask is not None:
                h_det = h_dq + (h_next - h_dq).detach()
                h_next = torch.where(ctx.mask[ctx.step], h_det, h_next)
            if ctx.is_last:
                halt = torch.ones_like(halt)
        # ponytail: the last step's halt is a constant (ones), so it carries no
        # grad — only backprop through it when it actually requires grad.
        outs, gouts = [h_next], [g_hnext]
        if halt.requires_grad:
            outs.append(halt)
            gouts.append(g_halt)
        torch.autograd.backward(outs, gouts, retain_graph=False)
        g_h = h_dq.grad
        g_x = x_dq.grad
        if ctx.hadamard:
            g_h = _HadamardTransform.apply(g_h)
            g_x = _HadamardTransform.apply(g_x)
        # ponytail: grad slots -- helix, h, x_enc(grad edge), xq/xs/xp(const),
        # engram_mem, step, prev_conf, mask, is_last, bits, hadamard.
        return (None, g_h, g_x, None, None, None, None, None, None, None, None, None, None)


class _BitnetAccumulate(torch.autograd.Function):
    """Accumulate the recurrent trajectory with int-storage of h_next.

    out = acc + halt_prob * h_next. The recurrent hidden states would otherwise
    pile up as bf16 across all n_loops (this is the only cost that scales with
    n_loops and blows VRAM at n_loops=48). We store h_next as int activation
    codes (tiny) instead of bf16, and free the bf16 tensor after the step.

    Forward values stay EXACT (the op runs on the live bf16 h_next); only the
    halt_prob gradient round-trips through the int codes (standard BitNet STE),
    so the recurrent-weight gradient path is unaffected.
    """

    @staticmethod
    def forward(ctx, acc, halt_prob, h_next, hq, hs, hp, bits, hadamard):
        ctx.bits, ctx.hadamard = bits, hadamard
        ctx.hp = hp
        # ponytail: store int codes (hq,hs) + the tiny halt_prob; never the bf16 h_next.
        ctx.save_for_backward(hq, hs, halt_prob)
        return acc + halt_prob * h_next

    @staticmethod
    def backward(ctx, g):
        hq, hs, halt_prob = ctx.saved_tensors
        h_dq = _dequant(hq, hs, ctx.bits, ctx.hp)
        g_acc = g
        g_halt = g * h_dq
        g_h_next = g * halt_prob
        return g_acc, g_halt, g_h_next, None, None, None, None, None


# ponytail: the loop body calls two custom Triton autograd.Functions
# (FusedGRULTI + FusedHFWBasis via DynamicFFN). Dynamo can't safely trace
# those, so run the loop eager; the surrounding anchor/synth still fuse.


class HelixCore(nn.Module):
    def __init__(self, d_model, max_loops=6, degree=6, num_frequencies=3, temperature=1.0,
                 lora_rank=8, use_checkpointing=True, max_sigma=1.0, rank=32,
                 dropbp=0.0, lcsb_ratio=0.0, sct_kernel=False, sct_fp8=False, fp8_kan=False,
                  bitnet_v2=False, bitnet_act_bits=8, bitnet_hadamard=True,
                  loop_checkpoint=False, mixture_of_depths=False, mod_capacity=0.5,
                  adaptive_loops=False, engram_vocab_size=65536):
        super().__init__()
        self.max_loops = max_loops
        self.dim = d_model
        self.use_checkpointing = use_checkpointing
        self.dropbp = dropbp
        self.bitnet_v2 = bitnet_v2
        self.loop_checkpoint = loop_checkpoint
        self.bitnet_act_bits = bitnet_act_bits
        self.bitnet_hadamard = bitnet_hadamard
        self.lcsb_ratio = lcsb_ratio
        self.mixture_of_depths = mixture_of_depths
        self.mod_capacity = mod_capacity
        self.adaptive_loops = adaptive_loops
        # ponytail: precompute all loop embeddings once — no dynamic torch.arange
        # / trig per step (avoids HBM allocs + fragmentation inside the recurrence).
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        embs = []
        for s in range(max_loops):
            si = s * inv_freq
            emb = torch.cat([torch.sin(si), torch.cos(si)], dim=-1)
            embs.append(emb.view(1, 1, d_model))
        self.register_buffer("loop_embs", torch.cat(embs, dim=0), persistent=False)
        # ponytail: MoD router is causal (sees only this token's h) so AR-safe.
        # Cheap Linear; routes top-k of capacity*B*T tokens into _loop_step per step.
        self.loop_router = nn.Linear(d_model, 1) if mixture_of_depths else None
        self.dynamic_ffn = DynamicFFN(d_model, degree, num_frequencies, temperature,
                                      rank=rank, max_sigma=max_sigma, fp8_kan=fp8_kan)
        # GRU gates -> SCTLinear: spectrally-binds the Core transition h_{t+1}=f(h_t,x)
        self.w_gate = SCTLinear(d_model, d_model * 3, rank=rank, max_sigma=max_sigma,
                                sct_kernel=sct_kernel, sct_fp8=sct_fp8,
                                bitnet_v2=bitnet_v2, bitnet_act_bits=bitnet_act_bits,
                                bitnet_hadamard=bitnet_hadamard)
        self.u_gate = SCTLinear(d_model, d_model * 3, rank=rank, max_sigma=max_sigma,
                                sct_kernel=sct_kernel, sct_fp8=sct_fp8,
                                bitnet_v2=bitnet_v2, bitnet_act_bits=bitnet_act_bits,
                                bitnet_hadamard=bitnet_hadamard)
        self.halt_predictor = nn.Linear(d_model, 1)
        # ponytail: slight negative halt bias — tokens start near full depth but
        # the predictor gets enough gradient signal (sigmoid ~0.12 at bias=-2 vs
        # ~0.018 at -4) to learn to early-exit within hundreds of steps.
        if adaptive_loops:
            nn.init.constant_(self.halt_predictor.bias, -2.0)
        self.lti = LTIInjection(d_model)
        self.relaxed_lora = DepthLoRA(d_model, rank=lora_rank, max_loops=max_loops)
        # loop-modulated LayerNorm (per-step learned scale/bias) -- preserved
        self.norm = LoopModulatedLN(d_model, max_loops)
        # mAR (Kimi/Moonshot Attention Residuals): learned pseudo-query pools the
        # recurrence trajectory by semantic direction, not vector magnitude.
        self.attn_res_query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.attn_res_norm = nn.LayerNorm(d_model)
        # Engram: conditional n-gram memory injected into the residual stream
        self.engram = DeepSeekEngram(d_model, engram_vocab_size=engram_vocab_size)

    @torch.compiler.disable
    def _loop_step(self, h, x_encoded, engram_mem, step, prev_conf):
        h_cond = h + self.loop_embs[step].to(h.dtype)
        # engram_mem is precomputed once in forward() (static across the loop)
        residual = h_cond + x_encoded + (engram_mem if engram_mem is not None else 0)
        h_norm = self.norm(residual, step)
        ffn_out, _ = self.dynamic_ffn(h_norm, confidence=prev_conf)
        ffn_out = ffn_out + self.relaxed_lora(h_norm, step)
        proj_x = self.w_gate(ffn_out)
        proj_h = self.u_gate(h)
        # Fused GRU gate + LTI injection in one Triton kernel (saves intermediate
        # HBM round-trips per loop). Keeps the recurrent transition spectrally bound.
        # ponytail: ARIA_NO_FUSE=1 falls back to the PyTorch reference (debug only).
        if os.environ.get("ARIA_NO_FUSE") == "1":
            xr, xz, xn = proj_x.chunk(3, dim=-1)
            hr, hz, hn = proj_h.chunk(3, dim=-1)
            r = torch.sigmoid(xr + hr)
            z = torch.sigmoid(xz + hz)
            n = torch.tanh(xn + r * hn)
            h_cand = (1 - z) * n + z * h
            h_next = self.lti.A_scale * h_cand + self.lti.B_scale * x_encoded
        else:
            h_next = FusedGRULTI.apply(proj_x, proj_h, h, x_encoded,
                                       self.lti.A_scale, self.lti.B_scale)
        halt_prob = torch.sigmoid(self.halt_predictor(h_next))
        return h_next, halt_prob

    def forward(self, x, tokens=None, max_loops=None, return_shallow_at=None):
        B, T, D = x.shape
        max_loops = max_loops or self.max_loops
        # ponytail: engram is keyed by raw token ids + n-gram stats; static across
        # the recurrence, so compute ONCE (not per-loop) to kill N-fold hash+lookup HBM traffic.
        engram_mem = self.engram(x, tokens) if tokens is not None else None
        # DropBP/LCSB: a fresh mask per forward (training only). masked steps detach
        # the recurrent gradient via h + (h_next - h).detach(), cutting BPTT depth.
        if self.training and (self.dropbp > 0 or self.lcsb_ratio > 0):
            mask = combined_mask(max_loops, self.dropbp, 1.0 - self.lcsb_ratio, x.device)
        else:
            mask = None
        # ponytail: full BitNet — quantize the input activation ONCE at the source
        # so the raw bf16 x is freed; every loop consumer gets the dequantized act.
        if self.bitnet_v2:
            from bitnet_v2 import bitnet_quantize_act
            x_encoded = bitnet_quantize_act(x, self.bitnet_act_bits, self.bitnet_hadamard)
        else:
            x_encoded = x
        if self.loop_checkpoint and self.training:
            # ponytail: whole-loop checkpoint -- the recurrence is recomputed in
            # backward, so the between-step h_next chain is NOT retained (true 4-bit
            # memory). Costs ~2x HelixCore compute; frees ~n_loops*(B,T,D).
            return checkpoint(self._run_loop, x_encoded, engram_mem, mask, max_loops,
                              False, return_shallow_at, use_reentrant=False)
        # ponytail: bitnet_v2 is quantization, NOT checkpointing -- do not conflate
        # them. The int-as-checkpoint (_BitnetRecStep) path is broken for multi-step
        # BPTT; use the plain path here (VRAM is fine without it -- see mod_bench).
        return self._run_loop(x_encoded, engram_mem, mask, max_loops,
                              False, return_shallow_at)

    def _run_loop(self, x_encoded, engram_mem, mask, max_loops, use_int_checkpoint,
                  return_shallow_at=None):
        """Recurrent chain + mAR.

        use_int_checkpoint=True -> per-step int-as-checkpoint (_BitnetRecStep +
        _BitnetAccumulate): within-step graph freed, but the between-step h_next
        chain is still retained (BPTT needs it) -> ~n_loops*(B,T,D) VRAM.
        use_int_checkpoint=False -> plain quantize + _loop_step; used inside the
        whole-loop torch.checkpoint path, where the entire loop is already
        recomputed so the chain needs no retention.
        """
        B, T, D = x_encoded.shape
        bits, had = self.bitnet_act_bits, self.bitnet_hadamard
        if self.bitnet_v2:
            xq, xs, xp = bitnet_int_codes(x_encoded, bits, had)
            x_dq = _dequant(xq, xs, bits, xp)
        else:
            x_dq = x_encoded
        h = x_encoded
        accumulated = torch.zeros_like(x_encoded)
        remaining_budget = torch.ones(B, T, 1, device=x_encoded.device, dtype=x_encoded.dtype)
        halt_probs = []
        shallow_state = None
        state_codes = []  # int codes of h_next per step (tiny); replaces bf16 states_history
        self.mod_aux_loss = torch.zeros((), device=x_encoded.device, dtype=x_encoded.dtype)
        halted = torch.zeros(B, T, 1, dtype=torch.bool, device=x_encoded.device)

        for step in range(max_loops):
            prev_conf = halt_probs[-1].detach() if len(halt_probs) > 0 else None
            is_last = (step == max_loops - 1)
            if self.adaptive_loops:
                active = ~halted                                      # (B,T,1)
                if active.all():
                    # ponytail: fast path — no token has halted yet (equiv. to
                    # non-adaptive). Avoids clone+gather/scatter overhead until
                    # some token actually exits early.
                    hq, hs, hp = bitnet_int_codes(h, bits, had)
                    h_dq = _dequant(hq, hs, hp, bits)
                    h_next, halt_prob = self._loop_step(h_dq, x_dq, engram_mem, step, prev_conf)
                    if not is_last:
                        halted = halt_prob > 0.5
                        halt_prob = torch.where(halted, torch.ones_like(halt_prob), halt_prob)
                else:
                    # ponytail: ACT — a token skips _loop_step once it "arrives"
                    # (halt_predictor conf > 0.5). Only active tokens run.
                    new_halted = halted.clone()
                    h_next = h.clone()
                    halt_prob = torch.zeros_like(h_next[..., :1])
                    if active.any():
                        idx = active.reshape(-1).nonzero(as_tuple=False).flatten()
                        h_f = h.reshape(-1, D); x_f = x_dq.reshape(-1, D)
                        h_a = h_f[idx].reshape(1, -1, D)
                        x_a = x_f[idx].reshape(1, -1, D)
                        pc = prev_conf.reshape(-1, 1)[idx].reshape(1, -1, 1) if prev_conf is not None else None
                        em = engram_mem.reshape(-1, D)[idx].reshape(1, -1, D) if engram_mem is not None else None
                        if self.bitnet_v2:
                            hq_a, hs_a, hp_a = bitnet_int_codes(h_a, bits, had)
                            h_a = _dequant(hq_a, hs_a, bits, hp_a)
                        h_next_a, halt_a = self._loop_step(h_a, x_a, em, step, pc)
                        conf = halt_a.reshape(-1, 1)
                        if is_last:
                            raw = torch.ones_like(conf)
                        else:
                            hard = conf > 0.5
                            raw = torch.where(hard, torch.ones_like(conf), conf)
                            new_halted.reshape(-1, 1)[idx] = hard
                        h_next.reshape(-1, D)[idx] = h_next_a.reshape(-1, D)
                        halt_prob.reshape(-1, 1)[idx] = raw
                    halted = new_halted
            elif self.mixture_of_depths:
                # ponytail: fixed capacity k=capacity*B*T tokens run _loop_step per
                # step; the rest carry h forward. k static -> static tensor sizes ->
                # GPU-friendly (Raposo et al. 2024). Router is causal; AR-safe.
                route = self.loop_router(h).squeeze(-1)              # (B,T)
                flat = route.reshape(-1)
                k = max(1, int(round(self.mod_capacity * flat.numel())))
                active = torch.topk(flat, k).indices                 # (k,)
                h_f = h.reshape(-1, D); x_f = x_dq.reshape(-1, D)
                # ponytail: gather active tokens into a (1,k,D) batch so the
                # 3D-shaped _loop_step (GRU/LTI/norm) runs unmodified, then scatter.
                h_a = h_f[active].reshape(1, k, D)
                x_a = x_f[active].reshape(1, k, D)
                pc = prev_conf.reshape(-1, 1)[active].reshape(1, k, 1) if prev_conf is not None else None
                em = engram_mem.reshape(-1, D)[active].reshape(1, k, D) if engram_mem is not None else None
                if self.bitnet_v2:
                    hq_a, hs_a, hp_a = bitnet_int_codes(h_a, bits, had)
                    h_a = _dequant(hq_a, hs_a, bits, hp_a)
                h_next_a, halt_a = self._loop_step(h_a, x_a, em, step, pc)
                h_next_a = h_next_a.reshape(k, D)
                halt_a = halt_a.reshape(k, 1)
                if is_last:
                    halt_a = torch.ones_like(halt_a)
                h_next = h.clone()
                h_next.reshape(-1, D)[active] = h_next_a
                halt_prob = torch.zeros_like(h_next[..., :1])
                halt_prob.reshape(-1, 1)[active] = halt_a
                p = torch.sigmoid(route)
                self.mod_aux_loss = self.mod_aux_loss + (
                    -(p * p.clamp_min(1e-7).log() + (1 - p) * (1 - p).clamp_min(1e-7).log())).mean()
            elif use_int_checkpoint:
                h_next, halt_prob = _BitnetRecStep.apply(
                    self, h, x_encoded, xq, xs, xp, engram_mem, step, prev_conf, mask,
                    is_last, bits, had)
            else:
                hq, hs, hp = bitnet_int_codes(h, bits, had)
                h_dq = _dequant(hq, hs, bits, hp)
                h_next, halt_prob = self._loop_step(h_dq, x_dq, engram_mem, step, prev_conf)
                if mask is not None:
                    h_det = h_dq + (h_next - h_dq).detach()
                    h_next = torch.where(mask[step], h_det, h_next)
                if is_last:
                    halt_prob = torch.ones_like(halt_prob)
            halt_prob = halt_prob * remaining_budget
            if self.bitnet_v2:
                # ponytail: store h_next as int codes; accumulate keeps the exact
                # bf16 forward value but frees the bf16 tensor (codes kept instead).
                hq, hs, hp = bitnet_int_codes(h_next, bits, had)
                state_codes.append((hq, hs, hp))
                if use_int_checkpoint:
                    accumulated = _BitnetAccumulate.apply(
                        accumulated, halt_prob, h_next, hq, hs, hp, bits, had)
                else:
                    accumulated = accumulated + halt_prob * h_next
            else:
                state_codes.append(h_next)
                accumulated = accumulated + halt_prob * h_next
            remaining_budget = remaining_budget - halt_prob
            halt_probs.append(halt_prob)
            if return_shallow_at is not None and step == (return_shallow_at - 1):
                shallow_state = accumulated.clone()
            h = h_next
            if not self.training and (remaining_budget < 1e-6).all():
                break

        # mAR: soft-pool the recurrence trajectory via a learned pseudo-query.
        # ponytail: we collect only the tiny (n_loops,B,T) step_logits, then fold the
        # weighted hidden states into one (B,T,D) accumulator -- never materializing
        # the (n_loops,B,T,D) buffer.
        if len(state_codes) > 1:
            step_logits = []
            for entry in state_codes:
                h_s = _dequant(entry[0], entry[1], bits, entry[2]) if self.bitnet_v2 else entry
                n_s = self.attn_res_norm(h_s)
                step_logits.append(torch.einsum("btd,d->bt", n_s, self.attn_res_query))
            step_w = F.softmax(torch.stack(step_logits, 0).unsqueeze(-1), dim=0)
            attn_res_accum = torch.zeros_like(x_encoded)
            for i, entry in enumerate(state_codes):
                h_s = _dequant(entry[0], entry[1], bits, entry[2]) if self.bitnet_v2 else entry
                attn_res_accum = attn_res_accum + step_w[i] * h_s
            final_output = 0.5 * accumulated + 0.5 * attn_res_accum
        else:
            final_output = accumulated
        return final_output, halt_probs, shallow_state
