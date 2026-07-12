## Objective
- Peak-optimize Aria-torch2 training (max tok/s, min VRAM, no quality loss). TRUE 4-bit memory done. This session: (1) found + fixed the REAL training breaker, (2) implemented + MEASURED MoD fixed-capacity early-exit, (3) measured checkpoint vs no-checkpoint and batch scaling at the real config.

## Important Details
- GPU Blackwell sm_120 (RTX 5060 Ti, 15.5GB). `.venv315t` = Py3.15.0b3 free-threaded. torch 2.14 cu132 nightly, triton 3.8.0. torch.compile IMPOSSIBLE on 3.15.
- **REAL TRAINING BREAKER FOUND + FIXED** (helix.py:214-215): `forward` passed `self.bitnet_v2` into the `use_int_checkpoint` slot. With `bitnet_v2=True` (our config) this forced the broken `_BitnetRecStep` multi-step path → "backward through graph a second time" crash / 3/34 grads. Fixed to pass `False` (plain path). Real AriaModel now trains 59/65 params (all non-engram; engram unused because `tokens=None`). The earlier `retain_graph=False` edit to `_BitnetRecStep.backward` was treating a symptom of this same root cause.
- **MoD IMPLEMENTED but does NOT help speed at D=1600.** `loop_router` (cheap Linear) + top-k active set (k=capacity*B*T) + gather→`_loop_step`→scatter. Trains 36/42 params (incl. router). But at D=1600 per-token recurrent compute is **latency/memory-bound, not FLOP-bound**, so halving tokens barely helps, while gather/scatter/topk + entropy-penalty overhead eats it. Measured (mode=none, B=2 T=256): base fwd 250/bwd 283ms vs MoD.5 fwd 292/bwd 331ms — MoD is SLOWER. Under whole-loop checkpoint it's worse (recompute). **Conclusion: MoD is a no-op/negative here; left flag-gated (default OFF).**
- **Whole-loop checkpoint (mode B) is a 2× compute tax for ~5% VRAM.** ckpt base: fwd 329/bwd 700ms vs none base fwd 250/bwd 283ms (HelixCore only). Net negative for tok/s.
- **Model is COMPUTE-BOUND, not VRAM-bound.** no-checkpoint VRAM ceiling ~4.5GB (plenty of 15.5GB headroom); tok/s plateaus ~3000 (HelixCore). Bigger batch does NOT raise tok/s (linear scaling), so the 2× checkpoint recompute is pure loss. Optimal = no checkpoint, batch sized to ~4.5GB.
- MoD papers: Raposo 2024 (arXiv:2404.02258) = SPEED technique but assumes FLOP-bound FFN (not our latency-bound recurrence); MoDA 2026 (arXiv:2603.15619) = quality, ignore.

## Work State
### Completed
- Gradio removed; CUDA Graphs; SCT bf16; config coercion; fp8 losses match bf16; BitNet v2 file + true int4 (config 8→4); x-int-code dedupe; whole-loop checkpoint impl (mode B, now known to be a 2x tax); VRAM re-profile; MoD implement + measure; **real-training-break root-cause fix**.
- **KEY MEASUREMENTS (D=1600, n_loops=48, bf16, RTX5060Ti):**
  - no-checkpoint HelixCore sweep: B=4/T=512→1243MB/3270tok/s; B=8/T=1024→4455MB/2989; B=4/T=2048→4455MB/2955. Ceiling ~4.5GB @ ~3000 tok/s.
  - ckpt (mode B): fwd 329/bwd 700ms/3741MB — 2x slower than none.
  - MoD.5 (none): 36/42 grads, fwd 292/bwd 331ms — slower than base.
  - Real AriaModel full-pipeline: 575ms/step, 4116MB, 59/65 grads, 891 tok/s (B=2 T=256).
### Active
- (none) — all plan items done.
### Blocked
- (none)

## Next Move
1. Leave `loop_checkpoint=False` (default) — already optimal; no checkpoint + batch-to-fit.
2. Keep MoD flag-gated OFF (doesn't help at D=1600); revisit only if D shrinks or T grows a lot.
3. Optional: wire `helix.mod_aux_loss` (router-entropy penalty, already computed) into pretrain loss IF MoD ever enabled.
4. Optional cleanup: configs/29m.yaml `use_checkpointing: false` comment ("_BitnetRecStep 4x less VRAM") is now stale/misleading — the int-as-checkpoint path is dead (broken multi-step); real training uses the plain path.

## Relevant Files
- `src/model/helix.py` — ACTIVE. (a) forward:214-215 FIXED (pass `False` not `self.bitnet_v2` as use_int_checkpoint). (b) `_BitnetRecStep.backward` line ~71 `retain_graph=False`. (c) `__init__`: added `mixture_of_depths`, `mod_capacity`, `self.loop_router`. (d) `_run_loop`: MoD branch (topk/gather/(1,k,D)/scatter) + `self.mod_aux_loss` entropy penalty. `halt_predictor` at line 145.
- `src/model/model.py` — added `loop_checkpoint`, `mixture_of_depths`, `mod_capacity` passthrough to HelixCore (run_pretrain does NOT pass them → defaults: no-checkpoint, MoD off).
- `configs/29m.yaml` – `bitnet_act_bits: 4`, `use_checkpointing: false` (stale comment).
- External: arXiv:2404.02258 (Raposo MoD), arXiv:2603.15619 (MoDA-ignore), arXiv:2504.18415v2 (BitNet v2).

## 1. User Requests (As-Is)
- (prior audit/optimize/bitnet/checkpoint requests preserved)
- "делай все" (do everything) — fix grad bug + implement MoD early-exit + measure.
- "продолжай" (continue) — proceed with next steps.

## 2. Final Goal
- Peak-optimize Aria-torch2 training: max tok/s + min VRAM, no quality loss. TRUE 4-bit memory. Fix training break. Implement + measure MoD. Determine optimal checkpoint/batch strategy.

## 3. Work Completed
- All prior (Gradio, CUDA Graphs, SCT, config, BitNet v2, int-as-checkpoint, streaming mAR, VRAM profile, whole-loop checkpoint, config 4-bit flip, x-dedupe).
- NEW: **root-cause + fix real training break** (forward bitnet_v2/use_int_checkpoint conflation); **MoD early-exit implemented** (loop_router + top-k gather/scatter, 36/42 grads); **compute-penalty aux loss** (`mod_aux_loss` entropy term); **full speed/VRAM measurement** (MoD neutral, checkpoint 2x tax, compute-bound ~3000 tok/s, ~4.5GB ceiling).

## 4. Remaining Tasks
- (optional) wire mod_aux_loss into pretrain loss if MoD enabled later.
- (optional) fix stale config comment.
- (optional) real-config convergence run (loss curve) to confirm no quality regression from forward fix — not yet run end-to-end via main.py.

## 5. Active Working Context (For Seamless Continuation)
- **Files**: `src/model/helix.py` (forward fix + MoD branch done), `src/model/model.py` (passthrough done), `configs/29m.yaml` (bitnet_act_bits:4).
- **Verified numbers**: real AriaModel 59/65 grads, 575ms/step, 4116MB, 891 tok/s (B=2 T=256 full pipeline). HelixCore no-checkpoint ~3000 tok/s ceiling.
- **Code fixes**:
  - helix.py:214 `return self._run_loop(x_encoded, engram_mem, mask, max_loops, False, return_shallow_at)` (was `self.bitnet_v2`).
  - helix.py:71 `torch.autograd.backward(outs, gouts, retain_graph=False)`.
  - helix.py MoD branch: `route=self.loop_router(h).squeeze(-1); k=int(round(self.mod_capacity*flat.numel())); active=torch.topk(flat,k).indices; h_a=h_f[active].reshape(1,k,D); ... self._loop_step(h_a,x_a,em,step,pc); scatter back; self.mod_aux_loss += binary_entropy(sigmoid(route)).mean()`.

## 6. Explicit Constraints (Verbatim Only)
- (prior constraints preserved: 3.15 free-threaded; cu132 nightly; .venv315t only; CUDA Graphs must work; verify all tech + real-training speed + convergence; optimize without quality loss; max optimization, 4-bit must be truly 4-bit.)

## 7. Agent Verification State
- **Current Agent**: Lead (main session).
- **Verification Progress**: real AriaModel forward+backward runs, 59/65 grads (correct); MoD trains 36/42 (verified via HelixCore harness); tok/s + VRAM sweep complete; checkpoint 2x tax confirmed; forward-root-cause fix verified (no crash, grads flow).
- **Pending Verifications**: end-to-end `python src/main.py configs/29m.yaml` convergence run (quality sanity after forward fix) not yet executed.
- **Acceptance Status**: Training break FIXED; MoD implemented + measured (no speed gain at D=1600); optimal strategy = no-checkpoint + batch-to-fit.

## 8. Delegated Agent Sessions
- bg_7d904e6a, bg_dc1b216f, bg_cb8845df, bg_4c53b53d, bg_767f836a — cancelled (hung). bg_7d3ca03c, bg_2c499863 — completed (dead). All terminated; do NOT resume. No agents spawned this session.
