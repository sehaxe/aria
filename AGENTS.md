# PROJECT KNOWLEDGE BASE — Aria (PyTorch)

**Stack:** PyTorch 2.14+cu132 · CUDA 13.0 · Triton 3.8.0 · FlexAttention · RTX 5060 Ti (sm_120, 16.7GB, 36 SMs) · Python 3.15.0b3 free-threaded

## OVERVIEW

Aria — 1.58-bit ternary LLM in PyTorch. Port of Rust/Burn `aria/`. GDN2 recurrent attention + HelixCore (48× loops) + SCTLinear (ternary low-rank) + FlexAttention + Triton kernels.

## STRUCTURE
```
src/
├── model/       # AriaModel, HelixCore, SCTLinear, Engram, ByteFlow, NSA, KAN
├── kernels/     # Triton kernels: sct_mm, sct_quant, fused_gru_lti, hfw_basis
├── optim/       # Muon (LOTUS), schedule_free, SCT-EggRoll
├── train/       # pretrain (MTP-4), SFT, GRPO, reward_models, controller
├── data/        # ByteFlowBinStreamer (mmap .bin), GRPO dataset
├── losses/      # VICReg
├── config.py    # AriaConfig from YAML
├── main.py      # Entry point
├── tui.py       # Terminal UI (thinking animation)
├── think.py     # LLM think loop logic
├── bitnet_v2.py # 4-bit int-as-checkpoint quant
└── spectral_tta.py # Test-time adaptation
configs/         # 29m.yaml
tests/           # smoke, compile, GRPO, SCT quant
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Model architecture | `src/model/model.py` | AriaModel: embed→Anchor→Helix(48×)→Synth(+CLD) |
| HelixCore (48 loops) | `src/model/helix.py` | GDN2 + KAN + gate + gradient checkpointing |
| SCTLinear (ternary) | `src/model/sct.py` | 1.58-bit low-rank linear, STE gradients |
| DeepSeekEngram | `src/model/engram.py` | 1M-vocab factual memory, phased warmup |
| ByteFlow tokenizer | `src/model/byteflow.py` | Byte-level BPE, 256 vocab |
| NSA attention | `src/model/nsa.py` | Native sparse: compress+select+window, gated |
| Phased training | `src/train/pretrain.py` | `train_phased()` — multi-stage declarative |
| SFT | `src/train/sft.py` | Supervised fine-tuning |
| GRPO | `src/train/grpo.py` | Group-relative policy optimization |
| Reward models | `src/train/reward_models.py` | Learned reward scoring |
| HymOpt | `src/optim/hymopt.py` | Muon + LOTUS factored momentum |
| EMA | `src/ema.py` | Shadow weight averaging |
| Mask | `src/mask.py` | LCSB + DropBP loop masking |
| QAC | `src/qac.py` | FP8/FP4 activation quant (STE) |
| CLD | `src/cld.py` | Deep - γ·shallow logit contrast (eval-only) |
| Spectral TTA | `src/spectral_tta.py` | 8 s-params, lr=0.05, 3ms/step |
| BitNet | `src/bitnet_v2.py` | 4-bit int-as-checkpoint, Hadamard de-rotate |
| Prompt compressor | `src/prompt_compressor.py` | Entropy-based compression |
| TUI | `src/tui.py` | Terminal thinking animation |
| Data pipeline | `src/data/dataset.py` | ByteFlowBinStreamer (mmap, zero-RAM shuffle) |
| Config | `configs/29m.yaml` | 598M param config (d_model=2048, n_heads=25, n_loops=48) |

## CONVENTIONS
- **SCTLinear** for all learned projections (ternary low-rank, rank=32)
- **FlexAttention** for AnchorBlock (2× causal self-attention)
- **Triton** for custom kernels — Python fallback via torch.compile (no CUDA Graphs: `nonzero()` in adaptive loops)
- **Muon** optimizer for 2D weights, **AdamW** for 1D/embed/embedding_table (embedding_table: lr=min(lr_adamw,2e-5), wd=0.0)
- **MTP-4** loss: weighted sum of 4 prediction heads (0.5^i decay)
- **Chunked GDN2**: C=64 chunks, O(D²) scan via element-wise ops
- **CLD**: eval-only, deep - γ·shallow logit subtraction (γ=0.1)
- **Gradient checkpointing**: every 4th helix loop
- **AMP**: bf16 autocast + GradScaler
- **Phased training**: declarative `training_stages` in YAML; stage 1 = engram_only freeze, stage 2 = full joint
- No `__init__.py` exports — imports by relative path
- No type hints on any function

## ANTI-PATTERNS
- No `as any`/`@ts-ignore` (this is Python, but same spirit: no `# type: ignore`)
- Empty catch blocks — never
- CUDA Graphs with adaptive loops — `nonzero()` breaks graph capture
- Weight decay on embedding_table — forces inactive embeddings toward zero
- `torch.jit.script` on `_scan` — use `torch.compile` fullgraph

## COMMANDS
```bash
source .venv315t/bin/activate
python src/main.py configs/29m.yaml              # Phased training (2 stages)
python src/main.py configs/29m.yaml --profile    # Profile mode
python -m pytest tests/test_smoke.py -xvs        # Smoke test
bash scripts/dump_code.sh                        # Dump all source to aria-code.md
```

## NOTES
- Blackwell sm_120 — RTX 5060 Ti, 16.7GB, 36 SMs
- torch.compile fullgraph works but torch.compile IMPOSSIBLE on Python 3.15 free-threaded
- NSA disabled by default (`nsa: true` in config)
- CLD enabled by default (`cld: true, cld_gamma: 0.1`)
- Gradient checkpointing on by default (every 4 loops)
- Engram init: `nn.init.normal_(weight, std=0.02)` — LLaMA/GPT standard, prevents bf16 overflow
- `.bin` data files mmap'd for zero-RAM global shuffle; falls back to synthetic generator
- `aria_tokenizer.so` removed from tracking (gitignored `*.so`)
- 133MB `checkpoint.pt` was gitignored and purged from history (fresh root commit)
