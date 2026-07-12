"""Config — YAML → dict, flat access."""
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any


@dataclass
class AriaConfig:
    d_model: int = 768
    n_heads: int = 12
    n_loops: int = 48
    vocab_size: int = 16384
    sct_rank: int = 32
    max_seq_len: int = 2048
    engram_vocab_size: int = 65536
    engram_only: bool = False
    training_stages: Optional[list] = None
    fp8_kan: bool = False
    inference_fp8: bool = False
    lr_muon: float = 2e-3
    lr_adamw: float = 3e-4
    wd_muon: float = 0.1
    wd_adamw: float = 0.01
    lotus_rank: int = 32
    auto_rank: bool = False
    beta_sf: float = 0.9
    lcsb_ratio: float = 0.0
    dropbp: float = 0.0
    qac_bits: int = 0
    # ponytail: BitNet v2 (arXiv:2504.18415) — Hadamard + 4/8-bit activation quant
    bitnet_v2: bool = False         # ponytail: apply ℋ-BitLinear activation quant in SCTLinear
    bitnet_act_bits: int = 8        # ponytail: stage 1 = 8 (train), stage 2 = 4 (continue)
    bitnet_hadamard: bool = True    # ponytail: online Hadamard before quant (the v2 trick)
    clip_norm: float = 1.0
    sct_l1: float = 1e-6
    grad_accum: int = 1
    log_every: int = 10
    max_seq_len: int = 2048
    batch_size: int = 8
    max_steps: int = 1000
    data_path: str = "data/train.bin"
    max_sigma: float = 1.0
    use_grpo: bool = False
    # --- performance / experimental flags (all OFF by default) ---
    sct_kernel: bool = False      # use fused Triton sct_mm matmul in SCTLinear
    sct_fp8: bool = False         # FP8/FP4 compute path in SCTLinear
    fa4: bool = False             # FlashAttention-4 (or flash backend) in NSAAttention
    use_cuda_graph: bool = False  # wrap pretrain step in a CUDAGraph
    use_checkpointing: bool = True  # gradient checkpointing in HelixCore loops
    adaptive_loops: bool = True    # data-dependent early-exit per token (free at init, learns to save)
    kv_cache: bool = False        # cached generation in GRPO rollout

    @classmethod
    def from_yaml(cls, path):
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)
        # PyYAML parses scientific-notation values like `1e-6` as strings; the
        # dataclass doesn't coerce types, so arithmetic (sct_l1 > 0) breaks.
        # Coerce string -> declared field type so config-driven training works.
        out = {}
        for k, v in d.items():
            if not hasattr(cls, k):
                continue
            if isinstance(v, str):
                ft = cls.__dataclass_fields__[k].type
                if ft is int:
                    try:
                        v = int(v)
                    except ValueError:
                        pass
                elif ft is float:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
            out[k] = v
        return cls(**out)
