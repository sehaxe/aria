"""GRPO post-training smoke test: reward engine + group-relative trainer."""
import sys
import torch

sys.path.insert(0, 'src')
from model.model import AriaModel
from train.grpo import GRPOTrainer
from train.pretrain import create_optimizer
from train.reward_models import VisualPrimitivesRewardEngine, decode_byte_trajectory
from data.grpo_dataset import GRPOMultimodalDataset, collate_grpo_fn
from torch.utils.data import DataLoader


def test_reward_engine():
    eng = VisualPrimitivesRewardEngine()
    good = "<think>We count the apples.</think><resp>There are 7 apples.</resp><|ref|>apple<|box|>[[10,20,30,40]]"
    assert eng.compute_format_reward(good) == 1.0
    assert eng.compute_quality_reward(good) == 1.0
    # answer must be the last number for counting to be scored
    assert abs(eng.compute_accuracy_counting(
        "<think>x</think><resp>7</resp>", 7) - 0.7) < 1e-6
    assert eng.compute_quality_reward("7 apples") == 0.0
    assert eng.compute_format_reward("<|ref|>x") < 1.0
    assert decode_byte_trajectory([256, 65, 257]) == "<think>A</think>"


def test_grpo_step():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AriaModel(d_model=256, n_heads=4, n_loops=4, rank=16, nsa=False, use_cld=False)
    model = model.to(dev).to(torch.bfloat16)
    ref = AriaModel(d_model=256, n_heads=4, n_loops=4, rank=16, nsa=False, use_cld=False)
    ref.load_state_dict(model.state_dict())
    ref = ref.to(dev).to(torch.bfloat16).eval()
    ref.requires_grad_(False)
    opts = create_optimizer(model)
    samples = [{"input_bytes": torch.randint(32, 127, (60,)).tolist(),
                "task_type": "counting", "target": 5} for _ in range(12)]
    ds = GRPOMultimodalDataset(samples, seq_len=8)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_grpo_fn)
    trainer = GRPOTrainer(model, ref, group_size=4, temperature=1.0)

    losses = []
    for i, batch in enumerate(loader):
        if i >= 2:
            break
        losses.append(trainer.train_step(batch, opts, clip=1.0))
    assert all(torch.isfinite(torch.tensor(l)) for l in losses), "NaN/inf GRPO loss"

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any((g.abs().sum() > 0).item() for g in grads), "no gradient flow"

    # Group must be non-degenerate: generated rewards vary within the group.
    patches, lengths, is_img, gt = next(iter(loader))
    patches = patches.to(dev).to(torch.bfloat16)
    lengths = lengths.to(dev)
    is_img = is_img.to(dev)
    with torch.no_grad():
        _, _, full = trainer._rollout(patches, lengths, is_img)
        rewards = trainer._reward(full, gt)
    assert rewards.std().item() > 0, "degenerate group -> no GRPO signal"


if __name__ == '__main__':
    test_reward_engine()
    test_grpo_step()
    print("PASS")
