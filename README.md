# Aria

1.58-bit ternary LLM in PyTorch. GDN2 recurrent attention + HelixCore + SCTLinear.

Port of the Rust/Burn `aria/` architecture with recurrent state-space attention.

## Quickstart

```bash
# Setup (Python 3.14 free-threaded)
uv venv -p 3.14
source .venv/bin/activate
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu132
uv pip install -e .

# Train (resumes automatically if checkpoint exists)
aria train configs/29m.yaml

# Generate
aria think --ckpt <path> --prompt "Hello"

# System check
aria doctor
```

## CLI

| Command | Description |
|---------|-------------|
| `aria train <config>` | Phased multi-stage training (engram → JEPA → joint) |
| `aria grpo <config>` | Group-relative policy optimization |
| `aria think` | Generate text from checkpoint |
| `aria doctor` | Report env, CUDA, VRAM, compile status |
| `aria tui` | Terminal UI (thinking animation) |
| `aria version` | Show version |

## Architecture (56.7M params)

```
Input → ByteFlow → HelixCore (8× GDN2 + DynamicFFN + NSA + gate)
         → DeepSeekEngram (65K-vocab factual memory)
         → AriaJEPA (world-model via masked latent prediction)
         → Synth → MTP-4 multi-token prediction
```

- **GDN2** — O(D²) recurrent delta-rule scan (chunked C=64)
- **SCTLinear** — 1.58-bit ternary low-rank (rank=16, STE gradients)
- **NSA** — Native sparse attention (compress+select+window)
- **Aria-JEPA** — I-JEPA masked latent + STP loop trajectory prediction
- **MTP-4** — 4-head multi-token prediction (0.5^i decay)
- **Muon (LOTUS)** — factored momentum optimizer for SCT weights
- **Spectral TTA** — 8-parameter test-time adaptation
- **Config-baked torch.compile** — `compile: helix` in YAML, no env vars

## Stack

| Component | Version |
|-----------|---------|
| PyTorch | 2.14+cu132 nightly |
| CUDA | 13.2 |
| Triton | 3.7+ |
| Python | 3.14 free-threaded |
| GPU | RTX 5060 Ti (sm_120, 16.7GB, 36 SMs) |
| Package | uv 0.11+ |

## Config

All hyperparameters in `configs/29m.yaml` — architecture, optimizer, data, stages.

## License

AGPL-3.0
