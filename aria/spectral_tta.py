import torch


def _collect_s_params(model):
    # SCTLinear stores its singular scales as `self.s` -> param name ends with '.s'
    return [p for n, p in model.named_parameters() if n.endswith('.s')]


@torch.enable_grad()
def spectral_tta_step(model, patches, lengths, is_img, steps=5, lr=0.05, persist=False):
    """Adapt SCTLinear singular scales (parameters named `.s`) on the prompt.

    The `s` vector in each SCTLinear weights the rank-1 components of the
    low-rank decomposition.  A few SGD steps on a self-prediction CE shift these
    singular values, adapting the model toward the current context.

    By default the original `s` values are restored at exit (side-effect-free) so
    the call can *measure* the adaptation gain.  Pass `persist=True` to keep the
    adapted values alive (e.g. for the generation that follows), then call
    `restore_spectral_tta(model)` once generation is done.

    Returns the final CE loss value (float).

    Overhead is small (~3ms/step on RTX 5060) — a handful of steps is enough.
    """
    s_params = _collect_s_params(model)
    if not s_params or steps == 0:
        return 0.0

    was_training = model.training
    prev_jepa = getattr(model, "jepa_active", False)
    model.train()
    # ponytail: skip JEPA/STP during the short adapt pass — its loss must not
    # enter the `s`-gradient, and jepa_only would early-return a scalar loss.
    model.jepa_active = False
    orig_data = [p.detach().clone() for p in s_params]
    if persist:
        model._spectral_tta_orig = orig_data

    loss_val = 0.0
    try:
        for _ in range(steps):
            for p in s_params:
                p.requires_grad_(True)
            out = model(patches, lengths, is_img, targets=patches)
            # AriaModel.forward returns a scalar or (loss, halt_probs[, jepa_loss])
            # depending on flags; the CE loss is always the first tensor element.
            loss = out if isinstance(out, torch.Tensor) else out[0]
            loss_val = float(loss.item())
            grads = torch.autograd.grad(loss, s_params, allow_unused=True)
            with torch.no_grad():
                for p, g in zip(s_params, grads):
                    if g is not None:
                        p.data.add_(g, alpha=-lr)
    finally:
        model.jepa_active = prev_jepa
        model.train(was_training)
        if not persist:
            with torch.no_grad():
                for p, orig in zip(s_params, orig_data):
                    p.data.copy_(orig)
    return loss_val


def restore_spectral_tta(model):
    """Revert `s` parameters previously adapted with `persist=True`."""
    orig = getattr(model, "_spectral_tta_orig", None)
    if orig is None:
        return
    s_params = _collect_s_params(model)
    with torch.no_grad():
        for p, o in zip(s_params, orig):
            p.data.copy_(o)
    del model._spectral_tta_orig
