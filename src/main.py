import argparse
import os
import time
import torch
from torch.utils.data import DataLoader

from model.model import AriaModel
from data.dataset import create_loader
from train.pretrain import create_optimizer, train, train_phased
from config import AriaConfig


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def run_pretrain(cfg, steps=None, checkpoint_path=None, save_every=500, resume_path=None, fresh=False):
    model = AriaModel(d_model=cfg.d_model, n_heads=cfg.n_heads, n_loops=cfg.n_loops,
                      rank=cfg.sct_rank, nsa=getattr(cfg, "nsa", False), nsa_every=cfg.nsa_every,
                      max_sigma=cfg.max_sigma,
                      sct_kernel=cfg.sct_kernel, sct_fp8=cfg.sct_fp8, fp8_kan=cfg.fp8_kan,
                      fa4=cfg.fa4, dropbp=cfg.dropbp, lcsb_ratio=cfg.lcsb_ratio,
                      use_checkpointing=cfg.use_checkpointing,
                      bitnet_v2=cfg.bitnet_v2, bitnet_act_bits=cfg.bitnet_act_bits,
                      bitnet_hadamard=cfg.bitnet_hadamard,
                      adaptive_loops=cfg.adaptive_loops,
                      engram_vocab_size=getattr(cfg, "engram_vocab_size", 65536),
                      jepa=getattr(cfg, "jepa", False),
                      jepa_pred_hidden=getattr(cfg, "jepa_pred_hidden", 1024),
                      jepa_context_keep=getattr(cfg, "jepa_context_keep", 0.7),
                      jepa_patch_size=getattr(cfg, "jepa_patch_size", 8),
                      jepa_lambda_k=getattr(cfg, "jepa_lambda_k", 1.0),
                      jepa_lambda_l=getattr(cfg, "jepa_lambda_l", 1.0),
                      jepa_stp=getattr(cfg, "jepa_stp", 0.1),
                      jepa_vicreg_var=getattr(cfg, "jepa_vicreg_var", 1.0),
                      jepa_vicreg_cov=getattr(cfg, "jepa_vicreg_cov", 1.0),
                      jepa_dropout=getattr(cfg, "jepa_dropout", 0.5),
                      jepa_kl_coef=getattr(cfg, "jepa_kl_coef", 0.01),
                       mtp=getattr(cfg, "mtp", True), mtp_k=getattr(cfg, "mtp_k", 4),
                       mtp_loss_coef=getattr(cfg, "mtp_loss_coef", 0.1),
                       loop_checkpoint=cfg.use_checkpointing)
    model = model.cuda().to(torch.bfloat16)
    # ponytail: torch.compile fuses the dense matmul/layernorm/attn subgraphs;
    # Triton SCT kernels + adaptive branching become graph breaks (fullgraph=False),
    # but fusion on the rest still wins. dynamic=True: shape varies by batch/seq.
    # ponytail: torch.compile is OFF by default — measured 6504 tok/s eager vs
    # 5569 compiled (graph breaks on every Triton SCT kernel + Hadamard
    # autograd.Function make fusion overhead exceed its win for this arch).
    # ARIA_COMPILE=1 → encoder/decoder only (experimental).
    # ARIA_COMPILE=2 → full model fullgraph=True (needs sct_kernel=false +
    # ARIA_NO_FUSE=1 + use_checkpointing=false for a clean graph).
    _compile_mode = os.environ.get("ARIA_COMPILE")
    if _compile_mode == "2":
        try:
            torch._dynamo.config.compiled_autograd = True
            # ponytail: compile only HelixCore (the dominant compute: 8x GDN2 +
            # FFN recurrence). Encoder/decoder + JEPA/NSA keep data-dependent
            # control flow (mask.sum() guards, gru rollout, engram lookup) that
            # breaks fullgraph; HelixCore alone is clean and fuses the hot loop.
            # dynamic=False: shapes are fixed per run (B,T,D static), avoids the
            # symbolic-stride inductor codegen bug ("Exponent must be non-negative").
            # fullgraph=False: NSA interleaves torch.nonzero() in the loop (adaptive
            # selection) which breaks fullgraph; the GDN2 recurrence still fuses,
            # only the sparse-attention node graph-breaks (few ops, negligible).
            model.helix = torch.compile(model.helix, fullgraph=False, dynamic=False)
            print("torch.compile: ON (helix, fullgraph=False)")
        except Exception as e:
            print(f"torch.compile: OFF ({e})")
    elif _compile_mode == "1":
        try:
            model.encoder = torch.compile(model.encoder, dynamic=True)
            model.decoder = torch.compile(model.decoder, dynamic=True)
            print("torch.compile: ON (encoder+decoder, experimental)")
        except Exception as e:
            print(f"torch.compile: OFF ({e})")
    print(f"Params: {count_params(model):,}")
    model.train()

    opts = create_optimizer(model, lr_muon=cfg.lr_muon, lr_adamw=cfg.lr_adamw,
                            wd_muon=cfg.wd_muon, wd_adamw=cfg.wd_adamw,
                            lotus_rank=cfg.lotus_rank,
                            engram_only=getattr(cfg, "engram_only", False))

    stages = getattr(cfg, "training_stages", None)
    if stages:
        train_phased(model, opts, stages,
                     batch_size=cfg.batch_size, seq_len=cfg.max_seq_len,
                     log_every=cfg.log_every,
                     use_cuda_graphs=cfg.use_cuda_graph,
                     sct_l1=cfg.sct_l1,
                     default_data_path=getattr(cfg, "data_path", None),
                     use_amp=False,
                     clip=getattr(cfg, "clip_norm", 1.0),
                     checkpoint_path=checkpoint_path,
                     save_every=save_every,
                     resume_path=resume_path,
                     fresh=fresh)
    else:
        loader = create_loader(batch_size=cfg.batch_size, seq_len=cfg.max_seq_len,
                               image_prob=0.5, data_path=getattr(cfg, "data_path", None))
        train(model, loader, opts, steps=steps if steps is not None else cfg.max_steps,
              use_cuda_graphs=cfg.use_cuda_graph, use_amp=False)
    print("OK")


def _synthetic_grpo_samples(n=16):
    samples = []
    for _ in range(n):
        n_bytes = torch.randint(20, 120, (1,)).item()
        # Keep bytes in the printable ASCII range so decoded text is non-empty.
        samples.append({
            "input_bytes": torch.randint(32, 127, (n_bytes,)).tolist(),
            "task_type": "counting",
            "target": int(torch.randint(1, 10, (1,)).item()),
        })
    return samples


def run_grpo(cfg, steps=None, kv_cache=False):
    from data.grpo_dataset import GRPOMultimodalDataset, collate_grpo_fn
    from train.grpo import GRPOTrainer

    model = AriaModel(d_model=cfg.d_model, n_heads=cfg.n_heads, n_loops=cfg.n_loops,
                      rank=cfg.sct_rank, nsa=getattr(cfg, "nsa", False), nsa_every=cfg.nsa_every,
                      max_sigma=cfg.max_sigma,
                      sct_kernel=cfg.sct_kernel, sct_fp8=cfg.sct_fp8, fp8_kan=cfg.fp8_kan,
                      fa4=cfg.fa4, dropbp=cfg.dropbp, lcsb_ratio=cfg.lcsb_ratio,
                      use_checkpointing=cfg.use_checkpointing,
                      bitnet_v2=cfg.bitnet_v2, bitnet_act_bits=cfg.bitnet_act_bits,
                      bitnet_hadamard=cfg.bitnet_hadamard,
                      adaptive_loops=cfg.adaptive_loops,
                      jepa=getattr(cfg, "jepa", False),
                      jepa_pred_hidden=getattr(cfg, "jepa_pred_hidden", 1024),
                      jepa_context_keep=getattr(cfg, "jepa_context_keep", 0.7),
                      jepa_patch_size=getattr(cfg, "jepa_patch_size", 8),
                      jepa_lambda_k=getattr(cfg, "jepa_lambda_k", 1.0),
                      jepa_lambda_l=getattr(cfg, "jepa_lambda_l", 1.0),
                      jepa_stp=getattr(cfg, "jepa_stp", 0.1),
                      jepa_vicreg_var=getattr(cfg, "jepa_vicreg_var", 1.0),
                      jepa_vicreg_cov=getattr(cfg, "jepa_vicreg_cov", 1.0),
                      jepa_dropout=getattr(cfg, "jepa_dropout", 0.5),
                      jepa_kl_coef=getattr(cfg, "jepa_kl_coef", 0.01),
                      speculative=getattr(cfg, "speculative", True),
                      speculative_k=getattr(cfg, "speculative_k", 4),
                       speculative_loss_coef=getattr(cfg, "speculative_loss_coef", 0.1),
                       mtp=getattr(cfg, "mtp", True), mtp_k=getattr(cfg, "mtp_k", 4),
                       mtp_loss_coef=getattr(cfg, "mtp_loss_coef", 0.1),
                       worldmodel_halt=getattr(cfg, "worldmodel_halt", True),
                       forecaster_loss_coef=getattr(cfg, "forecaster_loss_coef", 0.1))
    ref_model = AriaModel(d_model=cfg.d_model, n_heads=cfg.n_heads, n_loops=cfg.n_loops,
                          rank=cfg.sct_rank, nsa=getattr(cfg, "nsa", False), nsa_every=cfg.nsa_every,
                           max_sigma=cfg.max_sigma,
                          sct_kernel=cfg.sct_kernel, sct_fp8=cfg.sct_fp8, fp8_kan=cfg.fp8_kan,
                          fa4=cfg.fa4, dropbp=cfg.dropbp, lcsb_ratio=cfg.lcsb_ratio,
                          use_checkpointing=cfg.use_checkpointing,
                      bitnet_v2=cfg.bitnet_v2, bitnet_act_bits=cfg.bitnet_act_bits,
                      bitnet_hadamard=cfg.bitnet_hadamard,
                          adaptive_loops=cfg.adaptive_loops,
                      jepa=getattr(cfg, "jepa", False),
                      jepa_pred_hidden=getattr(cfg, "jepa_pred_hidden", 1024),
                      jepa_context_keep=getattr(cfg, "jepa_context_keep", 0.7),
                      jepa_patch_size=getattr(cfg, "jepa_patch_size", 8),
                      jepa_lambda_k=getattr(cfg, "jepa_lambda_k", 1.0),
                      jepa_lambda_l=getattr(cfg, "jepa_lambda_l", 1.0),
                      jepa_stp=getattr(cfg, "jepa_stp", 0.1),
                      jepa_vicreg_var=getattr(cfg, "jepa_vicreg_var", 1.0),
                      jepa_vicreg_cov=getattr(cfg, "jepa_vicreg_cov", 1.0),
                      jepa_dropout=getattr(cfg, "jepa_dropout", 0.5),
                      jepa_kl_coef=getattr(cfg, "jepa_kl_coef", 0.01),
                      speculative=getattr(cfg, "speculative", True),
                      speculative_k=getattr(cfg, "speculative_k", 4),
                       speculative_loss_coef=getattr(cfg, "speculative_loss_coef", 0.1),
                       mtp=getattr(cfg, "mtp", True), mtp_k=getattr(cfg, "mtp_k", 4),
                       mtp_loss_coef=getattr(cfg, "mtp_loss_coef", 0.1),
                       worldmodel_halt=getattr(cfg, "worldmodel_halt", True),
                       forecaster_loss_coef=getattr(cfg, "forecaster_loss_coef", 0.1))
    model = model.cuda().to(torch.bfloat16)
    ref_model.load_state_dict(model.state_dict())
    ref_model = ref_model.cuda().to(torch.bfloat16).eval()
    ref_model.requires_grad_(False)
    print(f"Params: {count_params(model):,}")
    model.train()

    opts = create_optimizer(model, lr_muon=cfg.lr_muon, lr_adamw=cfg.lr_adamw,
                            wd_muon=cfg.wd_muon, wd_adamw=cfg.wd_adamw,
                            lotus_rank=cfg.lotus_rank)
    ds = GRPOMultimodalDataset(_synthetic_grpo_samples(16), seq_len=8, max_patch_len=16)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_grpo_fn)

    trainer = GRPOTrainer(model, ref_model, group_size=4, beta=0.04, temperature=1.0,
                          kv_cache=kv_cache)

    n_steps = steps if steps is not None else cfg.max_steps
    for step, batch in enumerate(loader):
        if step >= n_steps:
            break
        loss = trainer.train_step(batch, opts, clip=1.0)
        if step % 5 == 0:
            print(f"grpo step {step}: loss={loss:.4f}")
    print("OK")


def main():
    ap = argparse.ArgumentParser(description="Aria training launcher")
    ap.add_argument("config_path", type=str, nargs="?", default=None, help="YAML config")
    ap.add_argument("--grpo", action="store_true", help="force GRPO post-training")
    ap.add_argument("--steps", type=int, default=None, help="override max_steps (non-phased)")
    ap.add_argument("--checkpoint", type=str, default=None, help="explicit checkpoint path")
    ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoint, start fresh")
    args = ap.parse_args()

    cfg = AriaConfig.from_yaml(args.config_path) if args.config_path else AriaConfig()
    use_grpo = args.grpo or getattr(cfg, "use_grpo", False)
    steps = args.steps
    ckpt_path = args.checkpoint or getattr(cfg, "checkpoint_path", None)
    resume_path = ckpt_path if (ckpt_path and os.path.exists(ckpt_path) and not args.fresh) else None

    if use_grpo:
        run_grpo(cfg, steps=steps, kv_cache=getattr(cfg, "kv_cache", False))
    else:
        run_pretrain(cfg, steps=steps, checkpoint_path=ckpt_path,
                     save_every=getattr(cfg, "save_every", 500),
                     resume_path=resume_path, fresh=args.fresh)


if __name__ == "__main__":
    main()
