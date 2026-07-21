import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class GRPOMultimodalDataset(Dataset):
    """Post-training GRPO dataset: (input patches, reward target metadata).

    Each sample yields raw byte patches (16 bytes each, padded to 768-dim
    encoder input) plus the metadata the reward engine needs to score the
    generated trajectory.
    """

    PAD_ID = 268

    def __init__(self, samples_list, seq_len=512, max_patch_len=16):
        self.samples = samples_list
        self.seq_len = seq_len
        self.max_patch_len = max_patch_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        raw_bytes = torch.tensor(sample["input_bytes"], dtype=torch.long)

        # Pad raw bytes to a multiple of 16 before reshaping into patches.
        if raw_bytes.numel() % 16 != 0:
            pad = 16 - (raw_bytes.numel() % 16)
            raw_bytes = F.pad(raw_bytes, (0, pad), value=self.PAD_ID)

        text_patches = raw_bytes.view(-1, 16)
        seq_len_actual = text_patches.shape[0]
        if seq_len_actual < self.seq_len:
            text_patches = F.pad(text_patches, (0, 0, 0, self.seq_len - seq_len_actual),
                                 value=self.PAD_ID)
        else:
            text_patches = text_patches[:self.seq_len]

        # 2D (seq_len,) image mask to match the encoder/decoder contract; the
        # collate stacks these into (B, seq_len).
        is_image_mask = torch.zeros(self.seq_len, dtype=torch.bool)

        # Pad each 16-byte patch up to the unified 768-dim encoder input.
        patches = F.pad(text_patches, (0, 768 - 16), value=self.PAD_ID).float()
        lengths = (text_patches != self.PAD_ID).sum(dim=-1).float()

        gt_metadata = {
            "task_type": sample["task_type"],  # 'counting' or 'maze'
            "target": sample["target"],         # int or list of [x, y]
        }
        return patches, lengths, is_image_mask, gt_metadata


def collate_grpo_fn(batch):
    """Stack tensors; keep the reward metadata list as-is (one dict per row)."""
    patches = torch.stack([item[0] for item in batch], dim=0)
    lengths = torch.stack([item[1] for item in batch], dim=0)
    is_image_mask = torch.stack([item[2] for item in batch], dim=0)
    gt_metadata = [item[3] for item in batch]
    return patches, lengths, is_image_mask, gt_metadata
