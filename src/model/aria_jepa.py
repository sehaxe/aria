import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model import AriaModel
from model.hwf_kan import HFW_KANLayer
from losses.vicreg import VICRegLoss


class AriaJEPA(nn.Module):
    """Aria-JEPA: latent-prediction self-supervision.

    Predicts the target network's latent state of a future patch instead of
    next-byte logits, training the encoder via VICReg in latent space. Target
    network is an EMA copy (BYOL/DINO style). Tokenizer-free: bytes enter via
    ByteFlowEncoder, not an embedding table.
    """

    def __init__(self, base_model: AriaModel, ema_decay=0.995, degree=6, num_frequencies=3):
        super().__init__()
        self.online_model = base_model
        self.ema_decay = ema_decay
        self.target_model = copy.deepcopy(base_model)
        # ponytail: deepcopy can drop the CUDA device in some torch builds, so pin
        # the EMA target to the online model's device/dtype explicitly.
        _p = next(base_model.parameters())
        self.target_model = self.target_model.to(_p.device, _p.dtype)
        for param in self.target_model.parameters():
            param.requires_grad = False
        self.predictor = HFW_KANLayer(base_model.d_model, base_model.d_model, degree, num_frequencies).to(_p.device, _p.dtype)
        self.vicreg_loss = VICRegLoss()

    @torch.no_grad()
    def update_target_network(self):
        for online_param, target_param in zip(self.online_model.parameters(), self.target_model.parameters()):
            target_param.data.mul_(self.ema_decay).add_(online_param.data, alpha=1.0 - self.ema_decay)

    def forward(self, patches, is_image_mask, patch_lengths=None):
        x_online = self.online_model.encoder(patches, is_image_mask)
        flat_tokens = patches.long()[:, :, 0]
        h_online, _, _ = self.online_model.helix(x_online, flat_tokens)

        with torch.no_grad():
            self.target_model.eval()
            x_target = self.target_model.encoder(patches, is_image_mask)
            h_target, _, _ = self.target_model.helix(x_target, flat_tokens)

        predicted_target = self.predictor(h_online[:, -1:])
        return self.vicreg_loss(predicted_target, h_target[:, 0:1])
