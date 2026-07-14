# Aria — src/model

Core model architecture (~2K lines).

## FILES
| File | Role |
|------|------|
| `model.py` | AriaModel: embed→HelixCore(48×)→Synth |
| `helix.py` | HelixCore — 48× GDN2 + KAN + gate + gradient checkpointing |
| `jepa.py` | AriaJEPA — I-JEPA masked latent prediction + Semantic Tube Prediction (STP) on the loop trajectory; two predictors (Knowledge=SwiGLU/SCTLinear, Logic=HFW-KAN) blended by a router, VICReg anti-collapse |
| `sct.py` | SCTLinear — 1.58-bit ternary low-rank, STE gradients |
| `engram.py` | DeepSeekEngram — 1M-vocab factual memory table |
| `byteflow.py` | ByteFlow — byte-level tokenizer |
| `nsa.py` | Native sparse attention (compress+select+window) |
| `hwf_kan.py` | Hyperbolic wavelet-filtered KAN |
| `loop_ln.py` | Layer norm per loop |
| `depth_lora.py` | Depth-wise LoRA adapters |
| `lti_injection.py` | LTI (linear time-invariant) state injection |

## KEY PATTERNS
- All learned projections use SCTLinear — no nn.Linear in model
- HelixCore: 48× GDN2 recurrent loops with DynamicFFN (Wav-KAN) FFN, NSA sparse-attention interleave, gate, and gradient checkpointing
- Engram: separate LR group (2e-5), no weight decay, init std=0.02
- Gradient checkpointing every 4th helix loop (config `grad_ckpt_every`)
- No `__init__.py` exports — relative imports only
- No type hints

## ANTI-PATTERNS (src/model only)
- Don't add nn.Linear — use SCTLinear
- Don't use `torch.jit.script` — use `torch.compile` fullgraph
- Don't remove gradient checkpointing without VRAM headroom check
