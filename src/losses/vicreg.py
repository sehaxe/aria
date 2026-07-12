import torch
import torch.nn as nn
import torch.nn.functional as F


class VICRegLoss(nn.Module):
    """
    VICReg (Variance-Invariance-Covariance Regularization).
    Prevents representation collapse without negative-sample memory cost.
    """

    def __init__(self, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0, threshold=1.0):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.threshold = threshold

    def forward(self, x, y):
        x = x.reshape(-1, x.size(-1))
        y = y.reshape(-1, y.size(-1))
        N, D = x.shape
        if N <= 1:
            return torch.tensor(0.0, device=x.device)

        sim_loss = F.mse_loss(x, y)

        x_centered = x - x.mean(dim=0, keepdim=True)
        y_centered = y - y.mean(dim=0, keepdim=True)

        std_x = torch.sqrt(x_centered.var(dim=0) + 1e-4)
        std_y = torch.sqrt(y_centered.var(dim=0) + 1e-4)
        std_loss_x = torch.mean(F.relu(self.threshold - std_x))
        std_loss_y = torch.mean(F.relu(self.threshold - std_y))
        std_loss = (std_loss_x + std_loss_y) / 2.0

        cov_x = (x_centered.T @ x_centered) / (N - 1)
        cov_y = (y_centered.T @ y_centered) / (N - 1)

        diag_mask = torch.eye(D, device=x.device, dtype=torch.bool)
        cov_loss_x = cov_x[~diag_mask].pow(2).sum() / D
        cov_loss_y = cov_y[~diag_mask].pow(2).sum() / D
        cov_loss = cov_loss_x + cov_loss_y

        return self.sim_coeff * sim_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss
