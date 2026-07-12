import torch, torch.nn.functional as F

def sft_loss(logits, targets, loss_mask):
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction='none')
    ce = ce.view_as(loss_mask)
    return (ce * loss_mask).sum() / loss_mask.sum().clamp(min=1)

def train_sft_step(model, batch, opts, clip=1.0):
    x, y, mask = batch
    x, y, mask = x.cuda(), y.cuda(), mask.cuda()
    for opt in opts:
        opt.zero_grad()
        if hasattr(opt, 'interpolate_params'):
            opt.interpolate_params()
    logits = model(x)
    loss = sft_loss(logits[0], y, mask)
    loss.backward()
    if clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    for opt in opts:
        opt.step()
    return loss.item()
