import argparse
import os
import time
import torch
from torch.utils.data import DataLoader

from aria.model.model import AriaModel
from aria.data.dataset import create_loader
from aria.train.pretrain import create_optimizer, train, train_phased
from aria.config import AriaConfig


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def build_model(cfg):
    """One factory: config -> AriaModel on CUDA, throughput knobs + compile applied.

    AriaModel ignores extra kwargs (its __init__ ends in **_), so we pass the
    union of pretrain/GRPO options; each path only consumes what it needs.
    """
    g = lambda k, d: getattr(cfg, k, d)
    model = AriaModel(
        d_model=cfg.d_model, n_heads=cfg.n_heads, n_loops=cfg.n_loops,
        rank=cfg.sct_rank, nsa=g("nsa", False), nsa_every=cfg.nsa_every,
        max_sigma=cfg.max_sigma, sct_kernel=cfg.sct_kernel, sct_fp8=cfg.sct_fp8,
        fp8_kan=cfg.fp8_kan, fa4=cfg.fa4, dropbp=cfg.dropbp, lcsb_ratio=cfg.lcsb_ratio,
        use_checkpointing=cfg.use_checkpointing, bitnet_v2=cfg.bitnet_v2,
        bitnet_act_bits=cfg.bitnet_act_bits, bitnet_hadamard=cfg.bitnet_hadamard,
        adaptive_loops=cfg.adaptive_loops, engram_vocab_size=g("engram_vocab_size", 65536),
        jepa=g("jepa", False), jepa_pred_hidden=g("jepa_pred_hidden", 1024),
        jepa_context_keep=g("jepa_context_keep", 0.7), jepa_patch_size=g("jepa_patch_size", 8),
        jepa_lambda_k=g("jepa_lambda_k", 1.0), jepa_lambda_l=g("jepa_lambda_l", 1.0),
        jepa_stp=g("jepa_stp", 0.1), jepa_vicreg_var=g("jepa_vicreg_var", 1.0),
        jepa_vicreg_cov=g("jepa_vicreg_cov", 1.0), jepa_dropout=g("jepa_dropout", 0.5),
        jepa_kl_coef=g("jepa_kl_coef", 0.01), speculative=g("speculative", True),
        speculative_k=g("speculative_k", 4), speculative_loss_coef=g("speculative_loss_coef", 0.1),
        mtp=g("mtp", True), mtp_k=g("mtp_k", 4), mtp_loss_coef=g("mtp_loss_coef", 0.1),
        worldmodel_halt=g("worldmodel_halt", True), forecaster_loss_coef=g("forecaster_loss_coef", 0.1),
        loop_checkpoint=cfg.use_checkpointing)
    model = model.cuda().to(torch.bfloat16)
    # ponytail: TF32 + cuDNN autotune + high-precision matmul (~20% on bf16 matmuls).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    # ponytail: compile the HelixCore recurrence (fullgraph=False because NSA's
    # torch.nonzero() breaks fullgraph; the GDN2 core still fuses). dynamic=False
    # since B,T,D are static per run. Gated by cfg.compile ("helix" / "").
    if g("compile", "") == "helix":
        torch._dynamo.config.compiled_autograd = True
        model.helix = torch.compile(model.helix, fullgraph=False, dynamic=False)
        print("torch.compile: ON (helix, fullgraph=False)")
    print(f"Params: {count_params(model):,}")
    return model


def run_pretrain(cfg, steps=None, checkpoint_path=None, save_every=500, resume_path=None, fresh=False):
    model = build_model(cfg)
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
              use_amp=False)
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
    from aria.data.grpo_dataset import GRPOMultimodalDataset, collate_grpo_fn
    from aria.train.grpo import GRPOTrainer

    model = build_model(cfg)
    ref_model = build_model(cfg)
    ref_model.load_state_dict(model.state_dict())
    ref_model = ref_model.eval()
    ref_model.requires_grad_(False)
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
