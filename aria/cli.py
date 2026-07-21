"""aria — command-line interface for the Aria ternary LLM.

Subcommands (ffmpeg-style: sane defaults, flags override config, progress to
stderr so stdout stays scriptable):

  aria train   <config.yaml> [--steps N] [--checkpoint P] [--fresh] [--resume P]
  aria grpo    <config.yaml> [--steps N]
  aria think   --ckpt P --prompt "text" [--config C] [--max-new N] [--temp F]
  aria data    pack <input_dir> -o <out.bin>
  aria doctor                       # env / cuda / compile / vram check
  aria tui                          # Textual control room
  aria version
"""
import argparse
import os
import sys

from aria import __version__


def _cmd_train(a):
    from aria.config import AriaConfig
    from aria.main import run_pretrain
    cfg = AriaConfig.from_yaml(a.config)
    ckpt = a.checkpoint or getattr(cfg, "checkpoint_path", None)
    resume = a.resume or (ckpt if (ckpt and os.path.exists(ckpt) and not a.fresh) else None)
    run_pretrain(cfg, steps=a.steps, checkpoint_path=ckpt,
                 save_every=getattr(cfg, "save_every", 500), resume_path=resume, fresh=a.fresh)


def _cmd_grpo(a):
    from aria.config import AriaConfig
    from aria.main import run_grpo
    cfg = AriaConfig.from_yaml(a.config)
    run_grpo(cfg, steps=a.steps, kv_cache=getattr(cfg, "kv_cache", False))


def _cmd_think(a):
    from aria.think import build_think_model, generate
    model, n_loops, device = build_think_model(config_path=a.config, checkpoint_path=a.ckpt)
    pairs = generate(model, a.prompt, max_new_bytes=a.max_new, temp=a.temp,
                     max_loops=n_loops, device=device)
    text = bytes(b for b, _ in pairs).decode("utf-8", errors="replace")
    sys.stdout.write(text + "\n")


def _cmd_data(a):
    if a.data_cmd == "pack":
        from aria.prepare_data import pack_text_folder_to_bin
        pack_text_folder_to_bin(a.input_dir, a.output)


def _cmd_doctor(a):
    import torch
    ok = True
    print(f"aria {__version__}")
    print(f"python {sys.version.split()[0]}")
    try:
        print(f"torch {torch.__version__}  cuda {torch.version.cuda}")
    except Exception as e:
        print(f"torch: MISSING ({e})"); ok = False
    if torch.cuda.is_available():
        n = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"gpu: {n}  sm_{cap[0]}{cap[1]}  {vram:.1f}GB")
    else:
        print("gpu: CUDA NOT AVAILABLE"); ok = False
    try:
        import triton
        print(f"triton {triton.__version__}")
    except Exception:
        print("triton: not installed (optional; compile path uses inductor)")
    try:
        m = torch.nn.Linear(8, 8).cuda()
        c = torch.compile(m)
        c(torch.randn(2, 8).cuda())
        print("torch.compile: OK")
    except Exception as e:
        print(f"torch.compile: BROKEN ({type(e).__name__})"); ok = False
    sys.exit(0 if ok else 1)


def _cmd_tui(a):
    from aria.tui import AriaTUI
    AriaTUI().run()


def build_parser():
    ap = argparse.ArgumentParser(prog="aria", description="Aria ternary LLM (1.58-bit, GDN2 + HelixCore + SCT)")
    ap.add_argument("-V", "--version", action="store_true", help="print version and exit")
    sub = ap.add_subparsers(dest="cmd")

    t = sub.add_parser("train", help="pretrain / phased training")
    t.add_argument("config", help="YAML config path")
    t.add_argument("--steps", type=int, default=None)
    t.add_argument("--checkpoint", default=None)
    t.add_argument("--resume", default=None, help="checkpoint to resume from")
    t.add_argument("--fresh", action="store_true", help="ignore existing checkpoint")
    t.set_defaults(fn=_cmd_train)

    g = sub.add_parser("grpo", help="GRPO post-training")
    g.add_argument("config")
    g.add_argument("--steps", type=int, default=None)
    g.set_defaults(fn=_cmd_grpo)

    th = sub.add_parser("think", help="generate from a checkpoint")
    th.add_argument("--ckpt", required=True)
    th.add_argument("--prompt", required=True)
    th.add_argument("--config", default=None)
    th.add_argument("--max-new", type=int, default=128)
    th.add_argument("--temp", type=float, default=0.7)
    th.set_defaults(fn=_cmd_think)

    d = sub.add_parser("data", help="data utilities")
    dsub = d.add_subparsers(dest="data_cmd")
    pk = dsub.add_parser("pack", help="pack .txt/.json/.md folder into a .bin")
    pk.add_argument("input_dir")
    pk.add_argument("-o", "--output", required=True)
    d.set_defaults(fn=_cmd_data)

    doc = sub.add_parser("doctor", help="check environment")
    doc.set_defaults(fn=_cmd_doctor)

    tu = sub.add_parser("tui", help="Textual control room")
    tu.set_defaults(fn=_cmd_tui)

    v = sub.add_parser("version", help="print version")
    v.set_defaults(fn=lambda a: print(__version__))
    return ap


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.version or getattr(a, "cmd", None) == "version":
        print(__version__)
        return 0
    if not getattr(a, "cmd", None):
        ap.print_help()
        return 0
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
