# PROJECT KNOWLEDGE BASE — Aria (PyTorch)

**Stack:** PyTorch 2.14+cu132 · CUDA 13.2 · Triton 3.7+ · RTX 5060 Ti (sm_120, 16.7GB, 36 SMs) · Python 3.14 free-threaded

## OVERVIEW

Aria — 1.58-bit ternary LLM in PyTorch. Port of Rust/Burn `aria/`. GDN2 recurrent attention + HelixCore (8× loops) + SCTLinear (ternary low-rank) + torch.compile.

## STRUCTURE
```
aria/
├── model/       # AriaModel, HelixCore, SCTLinear, Engram, ByteFlow, NSA, AriaJEPA
├── optim/       # Muon (LOTUS)
├── train/       # pretrain (MTP-4), GRPO, reward_models, controller
├── data/        # ByteFlowBinStreamer (mmap .bin), GRPO dataset
├── cli.py       # ffmpeg-style CLI (train/grpo/think/data/doctor/tui/version)
├── config.py    # AriaConfig from YAML
├── main.py      # build_model(cfg) factory + run_pretrain/run_grpo
├── tui.py       # Terminal UI (thinking animation + train monitor)
├── think.py     # LLM think loop logic
├── bitnet_v2.py # 4-bit activation quant & Hadamard generator
└── spectral_tta.py # Test-time adaptation
configs/         # 29m.yaml
tests/           # smoke, compile, GRPO, SCT quant
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Model architecture | `aria/model/model.py` | AriaModel: embed→HelixCore(8×)→Synth |
| HelixCore (8 loops) | `aria/model/helix.py` | GDN2 + DynamicFFN + gate (torch.compile fullgraph) |
| SCTLinear (ternary) | `aria/model/sct.py` | 1.58-bit low-rank linear, STE gradients |
| DeepSeekEngram | `aria/model/engram.py` | 65K-vocab factual memory, phased warmup |
| ByteFlow tokenizer | `aria/model/byteflow.py` | Byte-level BPE, 256 vocab |
| NSA attention | `aria/model/nsa.py` | Native sparse: compress+select+window, gated |
| Aria-JEPA | `aria/model/jepa.py` | I-JEPA masked latent + STP loop trajectory; dual predictor blended by router, VICReg anti-collapse |
| Phased training | `aria/train/pretrain.py` | `train_phased()` — multi-stage declarative |
| GRPO | `aria/train/grpo.py` | Group-relative policy optimization |
| HymOpt | `aria/optim/hymopt.py` | Muon + LOTUS factored momentum |
| EMA | `aria/ema.py` | Shadow weight averaging |
| Mask | `aria/mask.py` | LCSB + DropBP loop masking |
| Spectral TTA | `aria/spectral_tta.py` | 8 s-params, lr=0.05, 3ms/step |
| BitNet | `aria/bitnet_v2.py` | 4-bit activation quant, Kronecker-based Hadamard |
| CLI | `aria/cli.py` | Subcommands: train/grpo/think/data/doctor/tui/version |
| TUI | `aria/tui.py` | Terminal thinking animation + train monitor |
| Data pipeline | `aria/data/dataset.py` | ByteFlowBinStreamer (mmap, zero-RAM shuffle) |
| Config | `configs/29m.yaml` | 56.7M param (d_model=1152, n_heads=18, n_loops=8) |

## CONVENTIONS
- **SCTLinear** for all learned projections (ternary low-rank, rank=16)
- **torch.compile(helix, fullgraph=False, dynamic=False)** via `compile: helix` in YAML
- **Muon** optimizer for 2D weights, **AdamW** for 1D/embed/embedding_table (embedding_table: lr=min(lr_adamw,2e-5), wd=0.0)
- **MTP-4** loss: weighted sum of 4 prediction heads (0.5^i decay)
- **Chunked GDN2**: C=64 chunks, O(D²) scan via element-wise ops
- **Gradient checkpointing**: off by default (use_checkpointing: false)
- **AMP**: bf16 autocast + GradScaler
- **Phased training**: 3 stages — engram_only (freeze non-engram) → jepa_only (world-modeling, decoder frozen) → full joint (CE + JEPA)
- No type hints on any function

## ANTI-PATTERNS
- No `# type: ignore`
- Empty catch blocks — never
- CUDA Graphs with adaptive loops — `nonzero()` breaks graph capture
- Weight decay on embedding_table — forces inactive embeddings toward zero
- `torch.jit.script` on `_scan` — use `torch.compile` fullgraph

## COMMANDS
```bash
source .venv314t/bin/activate
aria train configs/29m.yaml              # Phased training (auto-resume)
aria train configs/29m.yaml --steps 10   # Smoke test
aria doctor                                # System check
aria think --ckpt <path> --prompt "Hi"    # Generate
aria tui                                   # Terminal UI
aria version                               # Show version
python -m pytest tests/test_smoke.py -xvs # Smoke test
```

## NOTES
- Blackwell sm_120 — RTX 5060 Ti, 16.7GB, 36 SMs
- torch.compile works on Python 3.14 free-threaded (GIL is re-enabled for triton load)
- NSA enabled by default (`nsa: true` in config)
- Gradient checkpointing off by default (`use_checkpointing: false` — perf > memory)
- Engram init: `nn.init.normal_(weight, std=0.02)` — LLaMA/GPT standard
- `.bin` data files mmap'd for zero-RAM shuffle; falls back to synthetic generator
- Checkpoint format: `{"model": state_dict, "global_step": N, "stage_idx": N, "stage_step": N}`
- Resumes automatically if checkpoint exists at `checkpoint_path` in config
