"""Cooperative training controller for the Gradio UI.

Runs Aria pretraining / GRPO in a background thread with pause, resume, stop
and periodic checkpointing. The existing run_pretrain/run_grpo in main.py have
no pause/resume/checkpoint, so this driver reuses the same building blocks
(create_optimizer, create_loader, GRPOTrainer) and adds cooperative control.

Log + metric lines are pushed to thread-safe buffers the UI polls; no stdout
redirection, so the Gradio server's own logs stay clean.
"""
import os
import time
import threading
from collections import deque
from pathlib import Path

import torch


class TrainController:
    def __init__(self, ckpt_dir="checkpoints"):
        self.lock = threading.Lock()
        self.logs = deque(maxlen=3000)
        self.metrics = []  # list of (step, loss, tok_s, mem)
        self.status = "idle"
        self.step = 0
        self.total_steps = 0
        self.params = {}
        self.thread = None
        self._pause = threading.Event()
        self._stop = threading.Event()
        self._ckpt_dir = Path(ckpt_dir)
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.profile_summary = ""
        self.profile_path = ""

    # ---- public control -------------------------------------------------
    def is_alive(self):
        return self.thread is not None and self.thread.is_alive()

    def configure(self, **kw):
        self.params = kw

    def start(self, resume=False):
        if self.is_alive():
            return False
        self._stop.clear()
        self._pause.clear()
        self.thread = threading.Thread(target=self._run, args=(resume,), daemon=True)
        self.thread.start()
        return True

    def pause(self):
        if self.is_alive():
            self._pause.set()

    def resume(self):
        self._pause.clear()

    def stop(self):
        self._stop.set()
        self._pause.clear()  # unblock a paused loop

    def profile(self, steps=10, trace_name="aria_profile"):
        """Profile a short training burst; exports a Chrome trace + top-ops summary.

        Runs inline (synchronous) because torch.profiler's CUDA backend (CUPTI)
        must execute on the thread that initialised CUDA — a background thread
        raises "External init callback must run in same thread".
        """
        self._profile_session(int(steps), trace_name)
        return True

    def get_profile(self):
        with self.lock:
            return self.profile_summary, self.profile_path

    def get_profile_text(self):
        s, p = self.get_profile()
        head = f"trace: {p}\n\n" if p else ""
        return (head + s) if s else (head or "no profile yet — press Profile")

    # ---- read state (UI polling) ---------------------------------------
    def get_logs(self):
        with self.lock:
            return "\n".join(self.logs)

    def get_metrics(self):
        with self.lock:
            return list(self.metrics)

    def get_state(self):
        with self.lock:
            frac = (self.step / self.total_steps) if self.total_steps else 0.0
            return {"status": self.status, "step": self.step,
                    "total": self.total_steps, "frac": frac}

    # ---- internals ------------------------------------------------------
    def _log(self, msg):
        with self.lock:
            self.logs.append(msg)

    def _add_metric(self, step, loss, tok_s, mem):
        with self.lock:
            self.metrics.append((step, loss, tok_s, mem))

    def _spin(self):
        """Honour pause/stop between steps. Returns False to abort the loop."""
        if self._stop.is_set():
            return False
        if self._pause.is_set():
            with self.lock:
                self.status = "paused"
            self._pause.wait()
        with self.lock:
            self.status = "running"
        return True

    def _ckpt_path(self, step):
        return self._ckpt_dir / f"aria_step_{step:06d}.pt"

    def _save(self, model, opts, step):
        sd = {"step": step, "model": model.state_dict(),
              "opts": [o.state_dict() for o in opts]}
        torch.save(sd, self._ckpt_path(step))
        self._log(f"[ckpt] saved step {step}")

    def _load_latest(self, model, opts):
        files = sorted(self._ckpt_dir.glob("aria_step_*.pt"))
        if not files:
            self._log("[ckpt] none found, starting fresh")
            return 0
        sd = torch.load(files[-1], map_location="cuda")
        model.load_state_dict(sd["model"])
        for o, osd in zip(opts, sd.get("opts", [])):
            o.load_state_dict(osd)
        self._log(f"[ckpt] resumed from {files[-1].name}")
        return sd.get("step", 0)

    def _build_model(self, p):
        from model.model import AriaModel
        m = AriaModel(d_model=p["d_model"], n_heads=p["n_heads"], n_loops=p["n_loops"],
                      rank=32,
                      nsa=p.get("nsa", False),
                      compile=p.get("compile", False),
                      max_sigma=p.get("max_sigma", 1.0),
                      sct_kernel=p.get("sct_kernel", False),
                      sct_fp8=p.get("sct_fp8", False),
                      fp8_kan=p.get("fp8_kan", False),
                      fa4=p.get("fa4", False),
                      dropbp=p.get("dropbp", 0.0),
                      lcsb_ratio=p.get("lcsb_ratio", 0.0),
                      bitnet_v2=p.get("bitnet_v2", False),
                      bitnet_act_bits=p.get("bitnet_act_bits", 8),
                      bitnet_hadamard=p.get("bitnet_hadamard", True))
        return m.cuda().to(torch.bfloat16)

    def _profile_session(self, steps, trace_name):
        from train.pretrain import create_optimizer
        try:
            p = self.params or {"d_model": 768, "n_heads": 12, "n_loops": 6,
                                 "batch_size": 2, "seq_len": 64, "lr_muon": 0.002,
                                 "lr_adamw": 0.0004, "image_prob": 0.5, "mode": "pretrain"}
            mode = p.get("mode", "pretrain")
            torch.cuda.empty_cache()
            model = self._build_model(p)
            if mode == "grpo":
                from data.grpo_dataset import GRPOMultimodalDataset, collate_grpo_fn
                from torch.utils.data import DataLoader
                from train.grpo import GRPOTrainer
                ref = self._build_model(p).eval()
                ref.load_state_dict(model.state_dict())
                ref.requires_grad_(False)
                opts = create_optimizer(model, lr_muon=p["lr_muon"], lr_adamw=p["lr_adamw"])
                ds = GRPOMultimodalDataset(self._synthetic_grpo_samples(p["batch_size"] * 4),
                                           seq_len=p["seq_len"], max_patch_len=16)
                loader = DataLoader(ds, batch_size=p["batch_size"], collate_fn=collate_grpo_fn)
                trainer = GRPOTrainer(model, ref, group_size=4, beta=0.04, temperature=1.0)
                it = iter(loader)
                model.train()

                def step_fn():
                    return trainer.train_step(next(it), opts, clip=1.0)
            else:
                from data.dataset import create_loader
                opts = create_optimizer(model, lr_muon=p["lr_muon"], lr_adamw=p["lr_adamw"])
                loader = create_loader(batch_size=p["batch_size"], seq_len=p["seq_len"],
                                       image_prob=p["image_prob"])
                it = iter(loader)
                model.train()

                def step_fn():
                    patches, lengths, is_img = next(it)
                    patches = patches.cuda().to(torch.bfloat16)
                    is_img = is_img.cuda()
                    for opt in opts:
                        opt.zero_grad()
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        if not is_img.all():
                            targets = patches[:, :, :16].long().cuda()
                            loss = model(patches, lengths, is_img, targets=targets)
                        else:
                            model(patches, lengths, is_img)
                            loss = None
                    if loss is not None:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        for opt in opts:
                            opt.step()
                    return float(loss.detach()) if loss is not None else 0.0

            step_fn()  # warmup (excluded from trace)
            trace_path = str(self._ckpt_dir / f"{trace_name}.json")
            # ponytail: one short trace, Chrome format + top-ops summary
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
            ) as prof:
                for _ in range(steps):
                    step_fn()
            torch.cuda.synchronize()
            prof.export_chrome_trace(trace_path)
            summary = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)
            self.profile_summary = summary
            self.profile_path = trace_path
            self._log(f"[profile] trace -> {trace_path}")
            self._log(f"[profile] top ops by CUDA time:\n{summary}")
        except Exception as e:
            self._log(f"[profile ERROR] {type(e).__name__}: {e}")
            import traceback
            self._log(traceback.format_exc())

    def _run(self, resume):
        try:
            if self.params.get("mode", "pretrain") == "grpo":
                self._run_grpo(resume)
            else:
                self._run_pretrain(resume)
        except Exception as e:  # surface errors in the log buffer
            with self.lock:
                self.status = "error"
            self._log(f"[ERROR] {type(e).__name__}: {e}")
            import traceback
            self._log(traceback.format_exc())
        finally:
            if self.status not in ("done", "stopped", "error"):
                with self.lock:
                    self.status = "idle"

    def _run_pretrain(self, resume):
        from data.dataset import create_loader
        from train.pretrain import create_optimizer

        p = self.params
        torch.cuda.empty_cache()
        model = self._build_model(p)
        opts = create_optimizer(model, lr_muon=p["lr_muon"], lr_adamw=p["lr_adamw"])
        loader = create_loader(batch_size=p["batch_size"], seq_len=p["seq_len"],
                               image_prob=p["image_prob"])
        start = self._load_latest(model, opts) if resume else 0
        with self.lock:
            self.step, self.total_steps = start, p["steps"]
        model.train()
        self._log(f"[pretrain] d_model={p['d_model']} n_loops={p['n_loops']} "
                  f"params={sum(x.numel() for x in model.parameters()):,}")
        t0 = time.time()
        it = iter(loader)
        for step in range(start, p["steps"]):
            if not self._spin():
                self._save(model, opts, self.step)
                with self.lock:
                    self.status = "stopped"
                self._log("[pretrain] stopped by user")
                return
            patches, lengths, is_img = next(it)
            patches = patches.cuda().to(torch.bfloat16)
            is_img = is_img.cuda()
            for opt in opts:
                opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if not is_img.all():
                    targets = patches[:, :, :16].long().cuda()
                    loss = model(patches, lengths, is_img, targets=targets)
                else:
                    model(patches, lengths, is_img)
                    loss = None
            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                for opt in opts:
                    opt.step()
                el = time.time() - t0
                tok_s = patches.numel() / max(0.001, el)
                mem = torch.cuda.max_memory_allocated() / 1e9
                with self.lock:
                    self.step = step + 1
                self._log(f"step {step}: loss={float(loss.detach()):.4f} tok/s={tok_s:.0f} mem={mem:.2f}GB")
                self._add_metric(step, float(loss.detach()), tok_s, mem)
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
            if (step + 1) % p["ckpt_every"] == 0:
                self._save(model, opts, step + 1)
        self._save(model, opts, p["steps"])
        with self.lock:
            self.status = "done"
        self._log("[pretrain] complete")

    def _synthetic_grpo_samples(self, n):
        samples = []
        for _ in range(n):
            n_bytes = torch.randint(20, 120, (1,)).item()
            samples.append({
                "input_bytes": torch.randint(32, 127, (n_bytes,)).tolist(),
                "task_type": "counting",
                "target": int(torch.randint(1, 10, (1,)).item()),
            })
        return samples

    def _run_grpo(self, resume):
        from data.grpo_dataset import GRPOMultimodalDataset, collate_grpo_fn
        from torch.utils.data import DataLoader
        from train.grpo import GRPOTrainer

        p = self.params
        torch.cuda.empty_cache()
        model = self._build_model(p)
        ref = self._build_model(p).eval()
        ref.load_state_dict(model.state_dict())
        ref.requires_grad_(False)
        opts = create_optimizer(model, lr_muon=p["lr_muon"], lr_adamw=p["lr_adamw"])
        ds = GRPOMultimodalDataset(self._synthetic_grpo_samples(p["batch_size"] * 4),
                                  seq_len=p["seq_len"], max_patch_len=16)
        loader = DataLoader(ds, batch_size=p["batch_size"], collate_fn=collate_grpo_fn)
        trainer = GRPOTrainer(model, ref, group_size=4, beta=0.04, temperature=1.0)
        start = self._load_latest(model, opts) if resume else 0
        with self.lock:
            self.step, self.total_steps = start, p["steps"]
        model.train()
        self._log(f"[grpo] d_model={p['d_model']} n_loops={p['n_loops']} "
                  f"params={sum(x.numel() for x in model.parameters()):,}")
        t0 = time.time()
        it = iter(loader)
        for step in range(start, p["steps"]):
            if not self._spin():
                self._save(model, opts, self.step)
                self.status = "stopped"
                self._log("[grpo] stopped by user")
                return
            batch = next(it)
            loss = trainer.train_step(batch, opts, clip=1.0)
            el = time.time() - t0
            tok_s = batch[0].numel() / max(0.001, el)
            mem = torch.cuda.max_memory_allocated() / 1e9
            with self.lock:
                self.step = step + 1
            self._log(f"grpo step {step}: loss={loss:.4f} tok/s={tok_s:.0f} mem={mem:.2f}GB")
            self._add_metric(step, loss, tok_s, mem)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            if (step + 1) % p["ckpt_every"] == 0:
                self._save(model, opts, step + 1)
        self._save(model, opts, p["steps"])
        with self.lock:
            self.status = "done"
        self._log("[grpo] complete")
