import torch, time
from optim.hymopt import Muon

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
    from model.sct import SCTLinear
    modules = [m for m in model.modules() if isinstance(m, SCTLinear)]
    if not modules:
        return None
    return sum(m.s.abs().sum() for m in modules)


# Per-model graph state. Keyed by id(model) so multiple models can coexist.
_GRAPH_STATE = {}


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


def train_step(model, batch, opts, clip=1.0, sct_l1=1e-6, scaler=None, use_cuda_graphs=False):
    patches, lengths, is_img, targets = _unpack_batch(batch)
    patches = patches.cuda(non_blocking=True)
    lengths = lengths.cuda(non_blocking=True)
    is_img = is_img.cuda(non_blocking=True)
    targets = targets.cuda(non_blocking=True)

    if not use_cuda_graphs:
        loss = _run_step_eager(model, patches, lengths, is_img, targets, opts, clip, sct_l1, scaler)
        return loss.item()

    # ---- CUDA graph path ----
    # ponytail: GradScaler capture inside a CUDAGraph is fragile (scaler.update()
    # touches host state); we require scaler=None for the graph path.
    if scaler is not None:
        raise RuntimeError("use_cuda_graphs=True requires scaler=None; "
                           "bf16 autocast without scaler works, or disable graphs.")

    key = id(model)
    state = _GRAPH_STATE.get(key)

    if state is None:
        # Warmup step in a side stream (required before capture per torch docs).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            _run_step_eager(model, patches, lengths, is_img, targets, opts, clip, sct_l1, None)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        # Static input buffers.
        static_patches = torch.empty_like(patches)
        static_lengths = torch.empty_like(lengths)
        static_is_img = torch.empty_like(is_img)
        static_targets = torch.empty_like(targets)
        static_patches.copy_(patches)
        static_lengths.copy_(lengths)
        static_is_img.copy_(is_img)
        static_targets.copy_(targets)

        graph = torch.cuda.CUDAGraph()
        static_loss = torch.empty((), device='cuda')

        with torch.cuda.graph(graph):
            for opt in opts:
                opt.zero_grad(set_to_none=False)  # keep grad tensors static
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = model(static_patches, static_lengths, static_is_img, targets=static_targets)
                if sct_l1 > 0:
                    l1 = _sct_l1(model)
                    if l1 is not None:
                        loss = loss + sct_l1 * l1
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            for opt in opts:
                opt.step()
            static_loss.copy_(loss.detach())

        state = {
            'graph': graph,
            'patches': static_patches,
            'lengths': static_lengths,
            'is_img': static_is_img,
            'targets': static_targets,
            'loss': static_loss,
            'clip': clip,
            'sct_l1': sct_l1,
        }
        _GRAPH_STATE[key] = state
        return static_loss.item()

    # Replay: copy new batch into static buffers, replay graph.
    if state['clip'] != clip or state['sct_l1'] != sct_l1:
        raise RuntimeError("clip / sct_l1 changed after graph capture; "
                           "invalidate by deleting the captured graph.")
    state['patches'].copy_(patches)
    state['lengths'].copy_(lengths)
    state['is_img'].copy_(is_img)
    state['targets'].copy_(targets)
    state['graph'].replay()
    return state['loss'].item()


def train(model, loader, opts, steps=1000, clip=1.0, sct_l1=0.0, log_every=10, use_amp=True,
          ema=None, use_cuda_graphs=False):
    model.train()
    t0 = time.time()
    # ponytail: graph path can't wrap a GradScaler; disable amp scaler when graphs on.
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if (use_amp and not use_cuda_graphs) else None
    for step, batch in enumerate(loader):
        if step >= steps:
            break
        loss = train_step(model, batch, opts, clip, sct_l1, scaler, use_cuda_graphs)
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
            print(f"step {step}: loss={loss:.4f} tok/s={tok_s:.0f} mem={mem:.2f}GB{halt_info}")
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()


def train_phased(model, opts, stages_config, batch_size=4, seq_len=64, log_every=10,
                 use_amp=True, use_cuda_graphs=False, sct_l1=0.0):
    """Run declarative multi-stage training with dynamic dataset switching."""
    model.train()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if (use_amp and not use_cuda_graphs) else None
    from data.dataset import create_loader

    global_step = 0
    for stage_idx, stage in enumerate(stages_config):
        stage_name = stage.get("name", f"stage_{stage_idx}")
        stage_steps = stage.get("steps", 100)
        stage_engram_only = stage.get("engram_only", False)
        stage_image_prob = stage.get("image_prob", 0.5)
        stage_data_path = stage.get("data_path", None)

        print(f"\n{'='*78}")
        print(f"=== [PHASED] Stage {stage_idx+1}/{len(stages_config)}: '{stage_name}' ===")
        print(f"    steps={stage_steps}  engram_only={stage_engram_only}"
              f"  image_prob={stage_image_prob}  data={stage_data_path or 'default'}")
        print(f"{'='*78}")

        # ponytail: for now create_loader ignores data_path (synthetic generator).
        # A real mmap ByteStreamer would read stage_data_path here.
        loader = create_loader(batch_size=batch_size, seq_len=seq_len,
                               image_prob=stage_image_prob, data_path=stage_data_path)

        # Set trainable params per stage
        active = 0
        for n, p in model.named_parameters():
            if stage_engram_only:
                p.requires_grad_(bool('engram' in n))
                if 'engram' in n:
                    active += p.numel()
                else:
                    p.grad = None
            else:
                p.requires_grad_(True)
                active += p.numel()
        print(f"[Phase] Trainable params: {active:,}")

        t0 = time.time()
        for step, batch in enumerate(loader):
            if step >= stage_steps:
                break
            loss = train_step(model, batch, opts, clip=1.0, sct_l1=sct_l1,
                              scaler=scaler, use_cuda_graphs=use_cuda_graphs)
            if global_step % log_every == 0:
                elapsed = time.time() - t0
                tok_s = batch[0].numel() * log_every / max(0.001, elapsed)
                mem = torch.cuda.max_memory_allocated() / 1e9
                print(f"[{stage_name}] global_step {global_step} "
                      f"(phase step {step}/{stage_steps}): "
                      f"loss={loss:.4f} tok/s={tok_s:.0f} mem={mem:.2f}GB")
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
            global_step += 1

    print(f"\n{'='*78}")
    print("=== [PHASED] All stages complete ===")
    print(f"{'='*78}")
