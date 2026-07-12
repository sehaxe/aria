import torch
import math


class SCT_EGGROLL_Optimizer:
    """
    Low-rank evolutionary (gradient-free) optimizer for Aria SCTLinear factors.
    Backprop-free learning at near-inference speed via population reward difference.
    """

    def __init__(self, model, lr=2e-3, noise_std=0.02, population_size=16, sct_rank=32):
        self.model = model
        self.lr = lr
        self.noise_std = noise_std
        self.pop_size = population_size
        self.sct_rank = sct_rank
        self.sct_layers = [m for m in model.modules() if type(m).__name__ == "SCTLinear"]

    @torch.no_grad()
    def step(self, dataloader, evaluation_reward_fn):
        orig_states = []
        for layer in self.sct_layers:
            orig_states.append((layer.U.clone(), layer.V.clone()))

        grad_U = [torch.zeros_like(l.U) for l in self.sct_layers]
        grad_V = [torch.zeros_like(l.V) for l in self.sct_layers]

        batch = next(iter(dataloader))
        scale_factor = self.noise_std / math.sqrt(self.sct_rank)

        for _ in range(self.pop_size):
            noise_list = []
            for i, layer in enumerate(self.sct_layers):
                eps_u = torch.randn_like(layer.U) * scale_factor
                eps_v = torch.randn_like(layer.V) * scale_factor
                noise_list.append((eps_u, eps_v))
                layer.U.copy_(orig_states[i][0] + eps_u)
                layer.V.copy_(orig_states[i][1] + eps_v)

            reward_pos = evaluation_reward_fn(self.model, batch)

            for i, layer in enumerate(self.sct_layers):
                eps_u, eps_v = noise_list[i]
                layer.U.copy_(orig_states[i][0] - eps_u)
                layer.V.copy_(orig_states[i][1] - eps_v)

            reward_neg = evaluation_reward_fn(self.model, batch)

            weight = reward_pos - reward_neg
            for i, layer in enumerate(self.sct_layers):
                eps_u, eps_v = noise_list[i]
                grad_U[i].add_(eps_u * weight)
                grad_V[i].add_(eps_v * weight)

        for i, layer in enumerate(self.sct_layers):
            step_U = (self.lr / (self.pop_size * self.noise_std)) * grad_U[i]
            step_V = (self.lr / (self.pop_size * self.noise_std)) * grad_V[i]
            layer.U.copy_(orig_states[i][0] + step_U)
            layer.V.copy_(orig_states[i][1] + step_V)
