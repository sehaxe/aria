# Aria-torch2

1.58-bit ternary LLM in PyTorch. GDN2 recurrent attention + HelixCore + SCTLinear + FlexAttention.

Port of the Rust/Burn `aria/` architecture with adaptive compute and recurrent state-space attention.

## Stack

| Component | Tech |
|-----------|------|
| Framework | PyTorch 2.14+cu132 nightly |
| GPU | Blackwell RTX 5060 Ti (sm_120, 16.7GB, 36 SMs) |
| Kernels | Triton 3.8.0 (gdn2_scan, sct_mm) |
| Python | 3.15.0b3 free-threaded |
| Attention | GDN2 (delta-rule recurrent scan) + FlexAttention (AnchorBlock) |
| Ternary | SCTLinear (1.58-bit low-rank, rank=32, STE gradients) |
| Optimizer | Muon (LOTUS factored momentum) + AdamW |

## Architecture

```
Input → Embed → AnchorBlock (2× FlexAttention + Wav-KAN FFN)
         → HelixCore (48× GDN2 + KAN + gate) → Synth → Output
```

- **GDN2** — O(D²) recurrent delta-rule scan (chunked C=64)
- **DeepSeekEngram** — 1M-vocab factual memory (mmap-streamed, phased warmup)
- **CLD** — contrastive shallow/deep logit subtraction (eval-only)
- **MTP-4** — 4-head multi-token prediction (0.5^i decay)
- **Phased Training** — declarative multi-stage config (engram warmup → joint fine-tuning)
- **Spectral TTA** — 8-parameter test-time adaptation (lr=0.05, 3ms/step)

## Setup

```bash
python3.15 -m venv .venv315t
source .venv315t/bin/activate
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu132
pip3 install -e .
```

## Train

```bash
python src/main.py configs/29m.yaml              # Phased training (2 stages)
python src/main.py configs/29m.yaml --profile     # Profile mode
python -m pytest tests/test_smoke.py -xvs         # Smoke test
```

## Config

All hyperparameters in `configs/29m.yaml` — architecture, optimizer, data, stages.

## Data

`prepare_data.py` packs `.txt`/`.json`/`.md` folders into `.bin` for mmap-based zero-copy streaming. Falls back to synthetic data when `.bin` absent.

## License

AGPL-3.0
