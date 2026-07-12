# PROJECT KNOWLEDGE BASE — Aria-torch2 (PyTorch)

**Stack:** PyTorch 2.13+cu130 · CUDA 13.0 · Triton 3.7.1 · FlexAttention · RTX 5060 Ti (sm_120, 16.7GB, 36 SMs)

## OVERVIEW

Aria — 1.58-bit ternary LLM in PyTorch. Port of Rust/Burn `aria/`. GDN2 recurrent attention + HelixCore (48× loops) + SCTLinear (ternary low-rank) + FlexAttention + Triton kernels.

## STRUCTURE
```
aria-torch2/
├── src/
│   ├── model/       # AriaModel, GDN2, HelixCore, SCTLinear, Anchor, KAN, Synth, FormatHead, NSA
│   ├── kernels/     # Triton kernels: gdn2_scan, sct_mm
│   ├── optim/       # Muon (LOTUS factored momentum)
│   ├── train/       # pretrain, SFT, GRPO loops
│   ├── data/        # ByteStreamer (mmap .bin)
│   ├── config.py    # AriaConfig from YAML
│   ├── ema.py       # EMA shadow weights
│   ├── mask.py      # LCSB + DropBP combined mask
│   ├── qac.py       # FP8/FP4 activation quantization
│   ├── cld.py       # CLD — contrastive shallow/deep logit subtraction
│   ├── sparse_68.py # 6:8 sparse mask utility
│   ├── forward_forward.py # Forward-Forward local learning
│   ├── prompt_compressor.py # Entropy-based prompt compression
│   ├── spectral_tta.py     # Spectral test-time adaptation
│   └── main.py      # Entry point
├── configs/         # 29m.yaml
└── tests/           # test_smoke.py
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Model architecture | `src/model/model.py` | AriaModel: embed→Anchor→Helix(48×)→Synth(+CLD) |
| GDN2 attention | `src/model/gdn2.py` | Delta-rule recurrent scan, O(D²) ops, torch.compiled |
| SCTLinear (ternary) | `src/model/sct.py` | 1.58-bit low-rank linear, STE gradients |
| HelixCore (48 loops) | `src/model/helix.py` | GDN2 + KAN + gate + gradient checkpointing |
| AnchorBlock | `src/model/anchor.py` | 2× FlexAttention + Wav-KAN FFN |
| NSA | `src/model/nsa.py` | Native sparse: compress+select+window, gated |
| HymOpt | `src/optim/hymopt.py` | Muon + LOTUS factored momentum |
| Training | `src/train/pretrain.py` | MTP-4 loss, bf16 AMP, GradScaler, EMA |
| EMA | `src/ema.py` | Shadow weight averaging |
| Mask | `src/mask.py` | LCSB + DropBP loop masking |
| QAC | `src/qac.py` | FP8/FP4 activation quant (STE) |
| CLD | `src/cld.py` | Deep - γ·shallow logit contrast |
| Spectral TTA | `src/spectral_tta.py` | SGD on s-vectors for test-time adaptation |
| Prompt Compressor | `src/prompt_compressor.py` | Entropy-based prompt compression |
| Forward-Forward | `src/forward_forward.py` | FFLinear with local goodness loss |
| Config | `configs/29m.yaml` | 29M param config (d_model=1600, n_heads=25, n_loops=48) |

## CONVENTIONS
- **SCTLinear** for all learned projections (ternary low-rank, rank=32)
- **FlexAttention** for AnchorBlock (2× causal self-attention)
- **Triton** for custom kernels (gdn2_scan, sct_mm — ponytail: Python fallback via torch.compile)
- **Muon** optimizer for 2D weights, **AdamW** for 1D/embed
- **MTP-4** loss: weighted sum of 4 prediction heads (0.5^i decay)
- **Chunked GDN2**: C=64 chunks, O(D²) scan via element-wise ops (not O(D³) full matmul)
- **CLD**: eval-only, deep - γ·shallow logit subtraction (γ=0.1)
- **Gradient checkpointing**: every 4th helix loop via `torch.utils.checkpoint`
- **AMP**: bf16 autocast + GradScaler
- **EMA**: optional, shadow weight decay=0.999

## ANTI-PATTERNS
- `h_acc` uninitialized in `sct_mm.py` (FIXED: now `h = tl.zeros(...)`)
- GDN2 Triton kernel backward was a stub (FIXED: proper backward on Triton path)
- `torch.jit.script` on `_scan` (FIXED: `torch.compile` fullgraph)
- No `__init__.py` exports — imports by relative path
- No type hints on any function

## COMMANDS
```bash
source .venv/bin/activate
python src/main.py configs/29m.yaml              # Train
python src/main.py configs/29m.yaml --profile    # Profile
python -m pytest tests/test_smoke.py -xvs        # Smoke test
```

## NOTES
- RTX 5060 Ti (sm_120, 16.7GB, 36 SMs) — Blackwell architecture
- PyTorch 2.13+cu130 — latest, FlexAttention built-in, torch.compile fullgraph
- Triton 3.7.1 — latest
- GDN2 scan: O(D²) element-wise, NOT O(D³) matmul (was using `unsqueeze @ unsqueeze` before fix)
- NSA not enabled by default (use `nsa: true` in config)
- CLD enabled by default (`cld: true, cld_gamma: 0.1`)
- Gradient checkpointing on by default (every 4 loops)
