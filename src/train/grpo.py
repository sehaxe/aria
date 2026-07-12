import torch
import torch.nn.functional as F
from .reward_models import VisualPrimitivesRewardEngine, decode_byte_trajectory

VOCAB_SIZE = 269
PAD_ID = 268
RESP_END = 267  # </resp> terminates the generated response


class GRPOTrainer:
    """Group Relative Policy Optimization without a value critic.

    For each batch element we draw `group_size` stochastic rollouts by
    autoregressively generating a reasoning trajectory (DBLT-style: the model
    predicts the 16 bytes of the next patch, which becomes the appended patch),
    score them with the reward engine, and normalize rewards within the group
    to get a relative advantage. The frozen `ref_model` provides the KL anchor.
    """

    def __init__(self, model, ref_model, group_size=4, beta=0.04, temperature=1.0,
                 max_new_patches=8, kv_cache=False):
        self.model = model
        self.ref_model = ref_model
        self.group_size = group_size
        self.beta = beta
        self.temperature = temperature
        self.max_new_patches = max_new_patches
        self.kv_cache = kv_cache
        self.engine = VisualPrimitivesRewardEngine()

    def _rollout(self, ctx_patches, ctx_lengths, ctx_is_img, kv_cache=False):
        """Generate a reasoning trajectory; return per-row log-prob under the
        online/reference policies plus the full token grid for reward scoring.
        """
        B = ctx_patches.shape[0]
        dev = ctx_patches.device
        patches, lengths, is_img = ctx_patches, ctx_lengths, ctx_is_img
        gen_logp_online = torch.zeros(B, device=dev)
        gen_logp_ref = torch.zeros(B, device=dev)
        gen_tokens = []
        active = torch.ones(B, dtype=torch.bool, device=dev)
        # ponytail: kv_cache caches the per-patch encoder outputs (encoder is
        # per-patch independent) and only encodes the newly appended patch each
        # step. The recurrent HelixCore is non-causal, so its state can't be
        # cached across appended patches — it still recomputes on the prefix.
        x_cache = rx_cache = None
        for _ in range(self.max_new_patches):
            if not active.any():
                break
            if kv_cache and x_cache is not None:
                x_new = self.model.encoder(patches[:, -1:], is_img[:, -1:])
                x_cache = torch.cat([x_cache, x_new], dim=1)
                logits, _ = self.model.run_encoded(x_cache, patches, is_img)
                r_new = self.ref_model.encoder(patches[:, -1:], is_img[:, -1:])
                rx_cache = torch.cat([rx_cache, r_new], dim=1)
                with torch.no_grad():
                    rlogits, _ = self.ref_model.run_encoded(rx_cache, patches, is_img)
            else:
                logits, _ = self.model(patches, lengths, is_img)
                with torch.no_grad():
                    rlogits, _ = self.ref_model(patches, lengths, is_img)
                if kv_cache:
                    x_cache = self.model.encoder(patches, is_img)
                    rx_cache = self.ref_model.encoder(patches, is_img)
            next_online = logits[:, -1]          # (B,16,269): predicts the next patch
            next_ref = rlogits[:, -1]
            probs = F.softmax(next_online / self.temperature, dim=-1)
            next_tok = torch.multinomial(probs.reshape(-1, VOCAB_SIZE), 1).reshape(B, 16)
            lp_o = F.log_softmax(next_online, -1).gather(-1, next_tok.unsqueeze(-1)).squeeze(-1).sum(-1)
            lp_r = F.log_softmax(next_ref, -1).gather(-1, next_tok.unsqueeze(-1)).squeeze(-1).sum(-1)
            gen_logp_online = gen_logp_online + lp_o * active.float()
            gen_logp_ref = gen_logp_ref + lp_r * active.float()
            gen_tokens.append(next_tok)
            active = active & (~(next_tok == RESP_END).any(dim=-1))
            new_patch = F.pad(next_tok, (0, 768 - 16), value=PAD_ID).float().to(patches.dtype)
            patches = torch.cat([patches, new_patch.unsqueeze(1)], dim=1)
            lengths = torch.cat([lengths, torch.full((B, 1), 16.0, device=dev)], dim=1)
            is_img = torch.cat([is_img, torch.zeros(B, 1, dtype=torch.bool, device=dev)], dim=1)

        ctx_bytes = ctx_patches[:, :, :16].long()
        if gen_tokens:
            gen_bytes = torch.stack(gen_tokens, 1)   # (B, G, 16)
            full = torch.cat([ctx_bytes, gen_bytes], 1)
        else:
            full = ctx_bytes
        return gen_logp_online, gen_logp_ref, full

    def _reward(self, full, gt_data):
        rewards = []
        for b in range(full.shape[0]):
            text = decode_byte_trajectory(full[b].reshape(-1).tolist())
            r_format = self.engine.compute_format_reward(text)
            r_quality = self.engine.compute_quality_reward(text)
            task_type = gt_data[b].get("task_type", "counting")
            if task_type == "counting":
                r_acc = self.engine.compute_accuracy_counting(text, gt_data[b]["target"])
            else:
                r_acc = self.engine.compute_accuracy_maze(text, gt_data[b]["target"])
            rewards.append(0.2 * r_format + 0.2 * r_quality + 0.6 * r_acc)
        return torch.tensor(rewards, device=full.device)

    def train_step(self, batch, opts, clip=1.0):
        patches, lengths, is_img, gt_data = batch
        patches, lengths, is_img = patches.cuda(), lengths.cuda(), is_img.cuda()

        for opt in opts:
            opt.zero_grad()
            if hasattr(opt, "interpolate_params"):
                opt.interpolate_params()

        group_log_probs, group_ref_log_probs, group_rewards = [], [], []
        for _ in range(self.group_size):
            lp_o, lp_r, full = self._rollout(patches, lengths, is_img, self.kv_cache)
            group_log_probs.append(lp_o)
            group_ref_log_probs.append(lp_r)
            group_rewards.append(self._reward(full, gt_data))

        log_probs = torch.stack(group_log_probs, 0)      # (G, B)
        ref_log_probs = torch.stack(group_ref_log_probs, 0)
        rewards = torch.stack(group_rewards, 0)          # (G, B)

        mean_r = rewards.mean(0, keepdim=True)
        std_r = rewards.std(0, keepdim=True) + 1e-8
        advantages = (rewards - mean_r) / std_r          # (G, B)

        ratio = torch.exp(log_probs - ref_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        kl = (log_probs - ref_log_probs).mean()          # KL(policy || ref)
        total_loss = policy_loss + self.beta * kl

        total_loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
        for opt in opts:
            opt.step()
        return total_loss.item()
