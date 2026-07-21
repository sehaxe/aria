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
from aria.mask import combined_mask
from aria.bitnet_v2 import bitnet_int_codes, _dequant, hadamard


class HelixCore(nn.Module):
    def __init__(self, d_model, max_loops=6, degree=6, num_frequencies=3, temperature=1.0,
                 lora_rank=8, use_checkpointing=True, max_sigma=1.0, rank=32,
                 dropbp=0.0, lcsb_ratio=0.0, sct_kernel=False, sct_fp8=False, fp8_kan=False,
                  bitnet_v2=False, bitnet_act_bits=8, bitnet_hadamard=True,
                  loop_checkpoint=False, mixture_of_depths=False, mod_capacity=0.5,
                   adaptive_loops=False, worldmodel_halt=False, nsa_attn=None, nsa_every=3,
                   forecaster_loss_coef=0.1, engram_vocab_size=65536):
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
        self.worldmodel_halt = worldmodel_halt
        self.forecaster_loss_coef = forecaster_loss_coef
        # B: world-model-grounded adaptive compute — an in-loop endpoint
        # forecaster + curvature-gated halt. Fused ONLY when adaptive_loops is
        # on, so the BLIND halt path (and non-adaptive tests) stay untouched.
        if adaptive_loops and worldmodel_halt:
            self.forecaster = SCTLinear(d_model, d_model, rank=rank, max_sigma=max_sigma,
                                        sct_kernel=sct_kernel, sct_fp8=sct_fp8,
                                        bitnet_v2=bitnet_v2, bitnet_act_bits=bitnet_act_bits,
                                        bitnet_hadamard=bitnet_hadamard)
            # 0 at init -> B starts identical to the existing halt_predictor,
            # then learns to halt earlier on low-curvature (converged) states.
            self.halt_curv_scale = nn.Parameter(torch.zeros(1))
        self.last_forecaster_loss = torch.zeros(())
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
        self.use_nsa = nsa_attn is not None
        self.nsa_attn = nsa_attn
        self.nsa_every = nsa_every

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
        # ponytail: pure-PyTorch GRU+LTI gate (no custom autograd.Function) so it
        # fuses under torch.compile fullgraph.
        xr, xz, xn = proj_x.chunk(3, dim=-1)
        hr, hz, hn = proj_h.chunk(3, dim=-1)
        r = torch.sigmoid(xr + hr)
        z = torch.sigmoid(xz + hz)
        n = torch.tanh(xn + r * hn)
        h_cand = (1 - z) * n + z * h
        h_next = self.lti.A_scale * h_cand + self.lti.B_scale * x_encoded
        halt_prob = torch.sigmoid(self.halt_predictor(h_next))
        return h_next, halt_prob

    def forward(self, x, tokens=None, max_loops=None, record_traj=False, compute_forecaster=True):
        B, T, D = x.shape
        max_loops = self.max_loops if max_loops is None else max_loops
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
            from aria.bitnet_v2 import bitnet_quantize_act
            x_encoded = bitnet_quantize_act(x, self.bitnet_act_bits, self.bitnet_hadamard)
        else:
            x_encoded = x
        if self.loop_checkpoint and self.training:
            # ponytail: whole-loop checkpoint -- the recurrence is recomputed in
            # backward, so the between-step h_next chain is NOT retained (true 4-bit
            # memory). Costs ~2x HelixCore compute; frees ~n_loops*(B,T,D).
            # ponytail: use_reentrant=True — the non-reentrant variant's
            # determinism check false-positives on this loop (custom Triton
            # autograd Functions + in-loop .detach()), so it raises
            # CheckpointError despite the recompute being exact. Reentrant
            # gives bit-identical gradients (verified vs no-checkpoint) and is
            # the right call here since torch.compile is unavailable on Py3.15.
            return checkpoint(self._run_loop, x_encoded, engram_mem, mask, max_loops,
                              False, record_traj, compute_forecaster,
                              use_reentrant=True)
        # ponytail: bitnet_v2 is quantization, NOT checkpointing -- do not conflate
        # them. The int-as-checkpoint (_BitnetRecStep) path is broken for multi-step
        # BPTT; use the plain path here (VRAM is fine without it -- see mod_bench).
        return self._run_loop(x_encoded, engram_mem, mask, max_loops,
                              False, record_traj, compute_forecaster)

    def _run_loop(self, x_encoded, engram_mem, mask, max_loops, use_int_checkpoint,
                  record_traj=False, compute_forecaster=True):
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
            # ponytail: de-rotate to original basis (see _BitnetRecStep.forward)
            if had:
                x_dq = hadamard(x_dq)
        else:
            x_dq = x_encoded
        h = x_encoded
        accumulated = torch.zeros_like(x_encoded)
        remaining_budget = torch.ones(B, T, 1, device=x_encoded.device, dtype=x_encoded.dtype)
        halt_probs = []
        state_codes = []  # int codes of h_next per step (tiny); replaces bf16 states_history
        self.mod_aux_loss = torch.zeros((), device=x_encoded.device, dtype=x_encoded.dtype)
        self.last_forecaster_loss = torch.zeros((), device=x_encoded.device, dtype=x_encoded.dtype)
        halted = torch.zeros(B, T, 1, dtype=torch.bool, device=x_encoded.device)
        b_active = self.adaptive_loops and self.worldmodel_halt
        prev_vel = None
        h_steps = []

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
                    h_dq = _dequant(hq, hs, bits, hp)
                    if had:
                        h_dq = hadamard(h_dq)
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
                            if had:
                                h_a = hadamard(h_a)
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
                    if had:
                        h_a = hadamard(h_a)
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
                if had:
                    h_dq = hadamard(h_dq)
                h_next, halt_prob = self._loop_step(h_dq, x_dq, engram_mem, step, prev_conf)
                if mask is not None:
                    h_det = h_dq + (h_next - h_dq).detach()
                    h_next = torch.where(mask[step], h_det, h_next)
                if is_last:
                    halt_prob = torch.ones_like(halt_prob)
            if b_active:
                vel = h_next - h
                if prev_vel is not None:
                    curv = (vel - prev_vel).norm(dim=-1, keepdim=True)
                else:
                    curv = torch.zeros_like(h_next[..., :1])
                # low curvature (converged) -> higher halt; scale learns from 0.
                hl = torch.logit(halt_prob.clamp(1e-4, 1 - 1e-4)) - self.halt_curv_scale * curv
                halt_prob = torch.sigmoid(hl)
                prev_vel = vel.detach()
                if self.training:
                    h_steps.append(h_next.clone())
            # ponytail: 3:1 interleave — one NSA self-attention pass into the
            # recurrent stream every `nsa_every` GDN2 steps (NSA mixes the hidden
            # across the sequence; GDN2 stays the backbone).
            if self.use_nsa and self.nsa_attn is not None and (step + 1) % self.nsa_every == 0:
                h_next = h_next + self.nsa_attn(h_next)
            halt_prob = halt_prob * remaining_budget
            if self.bitnet_v2:
                # ponytail: store h_next dequantized back to the ORIGINAL basis (not
                # the int tuple) so JEPA's semantic_tube can torch.stack the
                # trajectory; mAR below reads it directly without re-dequantizing.
                hq, hs, hp = bitnet_int_codes(h_next, bits, had)
                h_next_dq = _dequant(hq, hs, bits, hp)
                if had:
                    h_next_dq = hadamard(h_next_dq)
                state_codes.append(h_next_dq)
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
                n_s = self.attn_res_norm(entry)
                step_logits.append(torch.einsum("btd,d->bt", n_s, self.attn_res_query))
            step_w = F.softmax(torch.stack(step_logits, 0).unsqueeze(-1), dim=0)
            attn_res_accum = torch.zeros_like(x_encoded)
            for i, entry in enumerate(state_codes):
                attn_res_accum = attn_res_accum + step_w[i] * entry
            final_output = 0.5 * accumulated + 0.5 * attn_res_accum
        else:
            final_output = accumulated
        # ponytail: state_codes is the per-loop hidden trajectory [L,B,T,D];
        # returned (only when record_traj) for Semantic Tube Prediction.
        if b_active and self.training and compute_forecaster and len(h_steps) > 1:
            # B aux loss: predict the converged representation from each
            # intermediate hidden (the "where is my compute heading" signal).
            preds = self.forecaster(torch.stack(h_steps, 0))
            self.last_forecaster_loss = self.forecaster_loss_coef * F.mse_loss(
                preds, final_output.detach().unsqueeze(0).expand_as(preds))
        return final_output, halt_probs, state_codes
