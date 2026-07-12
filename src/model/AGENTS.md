# Aria — src/model

Core model architecture. 14 files, ~2K lines.

## FILES
| File | Role |
|------|------|
| `model.py` | AriaModel: embed→Anchor→HelixCore(48×)→Synth→CLD |
| `helix.py` | HelixCore — 48× GDN2 + KAN + gate + gradient checkpointing |
| `sct.py` | SCTLinear — 1.58-bit ternary low-rank, STE gradients |
| `engram.py` | DeepSeekEngram — 1M-vocab factual memory table |
| `byteflow.py` | ByteFlow — byte-level tokenizer |
| `nsa.py` | Native sparse attention (compress+select+window) |
| `kan.py` | Wav-KAN nonlinearity |
| `hwf_kan.py` | Hyperbolic wavelet-filtered KAN |
| `loop_emb.py` | Rotary position embeddings per loop |
| `loop_ln.py` | Layer norm per loop |
| `aria_jepa.py` | JEPA self-supervised learning head |
| `depth_lora.py` | Depth-wise LoRA adapters |
| `lti_injection.py` | LTI (linear time-invariant) state injection |

## KEY PATTERNS
- All learned projections use SCTLinear — no nn.Linear in model
- AnchorBlock: 2× FlexAttention + Wav-KAN FFN, then 48× HelixCore loops
- Engram: separate LR group (2e-5), no weight decay, init std=0.02
- Gradient checkpointing every 4th helix loop (config `grad_ckpt_every`)
- No `__init__.py` exports — relative imports only
- No type hints

## ANTI-PATTERNS (src/model only)
- Don't add nn.Linear — use SCTLinear
- Don't use `torch.jit.script` — use `torch.compile` fullgraph
- Don't remove gradient checkpointing without VRAM headroom check
