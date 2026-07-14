import argparse
import time
import torch
from torch.utils.data import DataLoader

from model.model import AriaModel
from data.dataset import create_loader
from train.pretrain import create_optimizer, train, train_phased
from config import AriaConfig


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def run_pretrain(cfg, steps=None):
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
                      mtp_loss_coef=getattr(cfg, "mtp_loss_coef", 0.1))
    model = model.cuda().to(torch.bfloat16)
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
                     use_amp=False)
    else:
        loader = create_loader(batch_size=cfg.batch_size, seq_len=cfg.max_seq_len, image_prob=0.5)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=str, nargs="?", default=None, help="YAML config")
    ap.add_argument("d_model", type=int, nargs="?", default=768)
    ap.add_argument("n_loops", type=int, nargs="?", default=6)
    ap.add_argument("steps", type=int, nargs="?", default=20)
    ap.add_argument("--config", dest="config_path", type=str, default=None,
                    help="YAML config (reads use_grpo)")
    ap.add_argument("--grpo", action="store_true", help="force GRPO post-training")
    args = ap.parse_args()

    if args.config_path is None and args.config not in (None, 768):
        args.config_path = args.config
    cfg = AriaConfig.from_yaml(args.config_path) if args.config_path else AriaConfig()
    use_grpo = args.grpo or getattr(cfg, "use_grpo", False)
    steps = args.steps if args.steps != 20 else None

    if use_grpo:
        run_grpo(cfg, steps=steps, kv_cache=getattr(cfg, "kv_cache", False))
    else:
        run_pretrain(cfg, steps=steps)


if __name__ == "__main__":
    main()
