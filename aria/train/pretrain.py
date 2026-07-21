import os, time, torch
from aria.optim.hymopt import Muon

def create_optimizer(model, lr_muon=2e-3, lr_adamw=3e-4, wd_muon=0.1, wd_adamw=0.01,
                     lotus_rank=32, engram_only=False):
    if engram_only:
        for n, p in model.named_parameters():
            if 'engram' in n:
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)
                p.grad = None
        print("[Phased] Frozen backbone — only Engram params trainable.")

    muon_p = []
    adamw_std_p = []
    adamw_embed_p = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'embedding_table' in n:
            adamw_embed_p.append(p)
        elif p.ndim == 2 and 'embed' not in n and 'head_out' not in n:
            muon_p.append(p)
        else:
            adamw_std_p.append(p)

    opts = []
    if muon_p:
        opts.append(Muon([{'params': muon_p, 'lr': lr_muon, 'wd': wd_muon, 'rank': lotus_rank}], lr=lr_muon))

    adamw_groups = []
    if adamw_std_p:
        adamw_groups.append({'params': adamw_std_p, 'lr': lr_adamw, 'weight_decay': wd_adamw})
    if adamw_embed_p:
        adamw_groups.append({'params': adamw_embed_p, 'lr': min(lr_adamw, 2e-5), 'weight_decay': 0.0})
    if adamw_groups:
        opts.append(torch.optim.AdamW(adamw_groups, capturable=True))

    return opts


def _unpack_batch(batch):
    """Batch may be (patches, lengths, is_img) or (patches, lengths, is_img, targets).
    If no targets, use patches as targets (self-prediction, shape-compatible)."""
    if len(batch) == 4:
        patches, lengths, is_img, targets = batch
    else:
        patches, lengths, is_img = batch
        targets = patches
    return patches, lengths, is_img, targets


def _sct_l1(model):
    from aria.model.sct import SCTLinear
    # Cache the SCTLinear list on first use (it's stable for the model's lifetime);
    # rescanning model.modules() every training step is pure overhead.
    modules = getattr(model, "_sct_modules", None)
    if modules is None:
        modules = [m for m in model.modules() if isinstance(m, SCTLinear)]
        model._sct_modules = modules
    if not modules:
        return None
    return sum(m.s.abs().sum() for m in modules)


def _run_step_eager(model, patches, lengths, is_img, targets, opts, clip, sct_l1, scaler):
    for opt in opts:
        opt.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        loss = model(patches, lengths, is_img, targets=targets)
        if sct_l1 > 0:
            l1 = _sct_l1(model)
            if l1 is not None:
                loss = loss + sct_l1 * l1
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    if clip > 0:
        if scaler is not None:
            for opt in opts:
                scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    if scaler is not None:
        for opt in opts:
            scaler.step(opt)
        scaler.update()
    else:
        for opt in opts:
            opt.step()
    return loss


def train_step(model, batch, opts, clip=1.0, sct_l1=1e-6, scaler=None):
    patches, lengths, is_img, targets = _unpack_batch(batch)
    patches = patches.cuda(non_blocking=True)
    lengths = lengths.cuda(non_blocking=True)
    is_img = is_img.cuda(non_blocking=True)
    targets = targets.cuda(non_blocking=True)
    # ponytail: return the loss TENSOR, not loss.item(). The .item() call forces a
    # full GPU sync every step, serializing the pipeline and ~2x-slowing tok/s.
    # Callers .item() only inside their log block (every log_every steps), so
    # steps pipeline freely between syncs.
    return _run_step_eager(model, patches, lengths, is_img, targets, opts, clip, sct_l1, scaler)


def train(model, loader, opts, steps=1000, clip=1.0, sct_l1=0.0, log_every=10, use_amp=True,
          ema=None):
    model.train()
    t0 = time.time()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None
    for step, batch in enumerate(loader):
        if step >= steps:
            break
        loss = train_step(model, batch, opts, clip, sct_l1, scaler)
        if ema:
            ema.update(model)
        if step % log_every == 0:
            elapsed = time.time() - t0
            tok_s = batch[0].numel() * log_every / max(0.001, elapsed)
            mem = torch.cuda.max_memory_allocated() / 1e9
            # Adaptive loops: how many tokens are halting?
            halt_info = ""
            h = model.helix
            if h.adaptive_loops and hasattr(h, 'halt_predictor') and h.halt_predictor is not None:
                with torch.no_grad():
                    hp = h.halt_predictor
                    bias = hp.bias.item() if hp.bias is not None else 0.0
                    conf = torch.sigmoid(torch.tensor(bias)).item()
                # Estimate halt rate from bias at init: bias=-4 → ~0.018 conf → ~0% halt
                # As training progresses, bias shifts positive → tokens start halting
                halt_info = f" halt_bias={bias:.2f} halt_conf={conf:.4f}"
            print(f"step {step}: loss={loss.item():.4f} tok/s={tok_s:.0f} mem={mem:.2f}GB{halt_info}")
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()


def _fmt_eta(sec):
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _save_ckpt(path, model, global_step, stage_idx, stage_step):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    sd = {
        "model": model.state_dict(),
        "global_step": global_step,
        "stage_idx": stage_idx,
        "stage_step": stage_step,
    }
    tmp = path + ".tmp"
    torch.save(sd, tmp)
    os.replace(tmp, path)


def _load_ckpt(path, model):
    # Only model weights are restored; the optimizer is rebuilt fresh on resume
    # (re-accrues momentum within a few steps). Loading Muon's custom float32
    # buffers through torch's generic optimizer state load corrupts their dtype.
    sd = torch.load(path, map_location="cuda", weights_only=False)
    model.load_state_dict(sd["model"])
    return sd["global_step"], sd["stage_idx"], sd["stage_step"]


def train_phased(model, opts, stages_config, batch_size=4, seq_len=64, log_every=10,
                 use_amp=True, sct_l1=0.0, default_data_path=None,
                 clip=1.0, checkpoint_path=None, save_every=500, resume_path=None, fresh=False):
    """Run declarative multi-stage training with dynamic dataset switching.

    Per-stage `batch_size` and `image_prob` override the global defaults so each
    phase can saturate the GPU (e.g. text-only engram/jepa stages take a bigger
    batch than the image-mixed joint stage). Checkpoints save model+optimizer
    state every `save_every` steps to `checkpoint_path`; pass `resume_path` (or a
    pre-existing file at `checkpoint_path`) to continue an interrupted run.
    """
    model.train()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None
    from aria.data.dataset import create_loader

    total_steps = sum(s.get("steps", 100) for s in stages_config)

    loaded = None
    if resume_path and os.path.exists(resume_path):
        loaded = resume_path
    elif checkpoint_path and os.path.exists(checkpoint_path) and not fresh:
        loaded = checkpoint_path

    global_step = 0
    start_stage = 0
    resume_phase_step = 0
    if loaded:
        global_step, start_stage, resume_phase_step = _load_ckpt(loaded, model)
        print(f"[Resume] loaded {loaded}: global_step={global_step} "
              f"stage={start_stage} phase_step={resume_phase_step}")

    run_start = time.time()
    ema_dt = None
    for stage_idx, stage in enumerate(stages_config):
        if stage_idx < start_stage:
            continue  # already finished before the resume point
        stage_name = stage.get("name", f"stage_{stage_idx}")
        stage_steps = stage.get("steps", 100)
        stage_engram_only = stage.get("engram_only", False)
        stage_jepa_only = stage.get("jepa_only", False)
        stage_image_prob = stage.get("image_prob", 0.5)
        stage_data_path = stage.get("data_path", default_data_path)
        stage_batch = stage.get("batch_size", batch_size)

        # Aria-JEPA: enable world-model training in non-engram stages; in a
        # jepa_only stage the decoder is frozen and only the JEPA loss trains
        # the encoder/helix/predictors.
        jepa_on = getattr(model, "jepa", None) is not None
        model.jepa_only = bool(jepa_on and stage_jepa_only)
        model.jepa_active = bool(jepa_on and not stage_engram_only)

        print(f"\n{'='*78}")
        print(f"=== [PHASED] Stage {stage_idx+1}/{len(stages_config)}: '{stage_name}' ===")
        print(f"    steps={stage_steps}  batch={stage_batch}  engram_only={stage_engram_only}"
              f"  image_prob={stage_image_prob}  data={stage_data_path or 'default'}")
        print(f"{'='*78}")

        loader = create_loader(batch_size=stage_batch, seq_len=seq_len,
                               image_prob=stage_image_prob, data_path=stage_data_path)
        loader_iter = iter(loader)

        # Set trainable params per stage
        active = 0
        for n, p in model.named_parameters():
            if stage_engram_only:
                p.requires_grad_(bool('engram' in n))
                if 'engram' in n:
                    active += p.numel()
                else:
                    p.grad = None
            elif stage_jepa_only:
                # Aria-JEPA world-modeling: freeze the generative decoder; train
                # encoder + HelixCore + both predictors by the JEPA/STP loss.
                p.requires_grad_(not ('decoder' in n))
                if 'decoder' in n:
                    p.grad = None
                else:
                    active += p.numel()
            else:
                p.requires_grad_(True)
                active += p.numel()
        print(f"[Phase] Trainable params: {active:,}")

        t0 = time.time()
        start_phase = resume_phase_step if (stage_idx == start_stage and loaded) else 0
        for step in range(start_phase, stage_steps):
            batch = next(loader_iter)
            loss = train_step(model, batch, opts, clip=clip, sct_l1=sct_l1,
                              scaler=scaler)
            if global_step % log_every == 0:
                elapsed = time.time() - t0
                dt = elapsed / max(1, log_every)
                ema_dt = dt if ema_dt is None else 0.9 * ema_dt + 0.1 * dt
                # Real token throughput: B*T patches (not B*T*768 patch-dim).
                nb, nt = batch[0].shape[0], batch[0].shape[1]
                tok_s = nb * nt * log_every / max(0.001, elapsed)
                mem = torch.cuda.max_memory_allocated() / 1e9
                remaining = total_steps - global_step
                eta = _fmt_eta(remaining * ema_dt)
                jaux = getattr(model, "last_jepa_aux", {}) or {}
                jstr = ""
                if jaux:
                    jstr = " " + " ".join(f"{k}={v:.4f}" for k, v in jaux.items())
                print(f"[{stage_name}] global_step {global_step} "
                      f"(phase step {step}/{stage_steps}): "
                      f"loss={loss.item():.4f} tok/s={tok_s:.0f} mem={mem:.2f}GB "
                      f"eta={eta}{jstr}")
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
            global_step += 1
            if checkpoint_path and global_step % save_every == 0:
                _save_ckpt(checkpoint_path, model, global_step, stage_idx, step + 1)

    if checkpoint_path:
        # stage_idx == len(stages) marks a finished run; on resume the stage
        # loop skips everything and prints "All stages complete" immediately.
        _save_ckpt(checkpoint_path, model, global_step, len(stages_config), 0)
    print(f"\n{'='*78}")
    print(f"=== [PHASED] All stages complete ({global_step} steps, "
          f"{time.time() - run_start:.0f}s) ===")
    print(f"{'='*78}")
