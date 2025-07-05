import numpy as np
import torch
import torch.nn as nn
from collections import deque
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm
from harl.algorithms.actors.on_policy_base import OnPolicyBase


class HAPPO_Potential(OnPolicyBase):
    def __init__(self, args, obs_space, act_space, num_agents, device=torch.device("cpu")):
        super(HAPPO_Potential, self).__init__(args, obs_space, act_space, device)

        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]

        self.potential_weight = args.get("potential_weight", 0.25)
        self.coop_bonus_weight = args.get("coop_bonus_weight", 0.15)
        self.int_rnd_weight = args.get("int_rnd_weight", 0.1)
        self.num_agents = num_agents

        local_obs_dim = int(np.prod(obs_space.shape))
        global_obs_dim = local_obs_dim * num_agents

        self.rnd_target = nn.Sequential(
            nn.Linear(global_obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128)
        )
        for p in self.rnd_target.parameters():
            p.requires_grad = False

        self.rnd_predictor = nn.Sequential(
            nn.Linear(global_obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.rnd_optimizer = torch.optim.Adam(self.rnd_predictor.parameters(), lr=1e-4)
        self._rnd_err_queue = deque(maxlen=1000)
        self.rnd_target = self.rnd_target.to(self.device)
        self.rnd_predictor = self.rnd_predictor.to(self.device)

    def update(self, sample):
        if len(sample) == 10:
            (
                obs_batch,
                next_obs_batch,
                rnn_states_batch,
                actions_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                available_actions_batch,
                factor_batch,
            ) = sample
        elif len(sample) == 9:
            (
                obs_batch,
                rnn_states_batch,
                actions_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                available_actions_batch,
                factor_batch,
            ) = sample
            next_obs_batch = None
        else:
            raise ValueError(f"Unexpected sample length: {len(sample)}")

        obs_batch = check(obs_batch).to(**self.tpdv)
        next_obs_batch = check(next_obs_batch).to(**self.tpdv) if next_obs_batch is not None else None
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        factor_batch = check(factor_batch).to(**self.tpdv)
        available_actions_batch = check(available_actions_batch).to(**self.tpdv)

        # 构造中心化观测
        batch_size = obs_batch.shape[0]
        local_dim = obs_batch.shape[1]
        obs_flat = obs_batch.reshape(batch_size, -1)
        centralized_obs = torch.cat([obs_flat] * self.num_agents, dim=-1)

        if next_obs_batch is not None:
            next_flat = next_obs_batch.reshape(batch_size, -1)
            centralized_next_obs = torch.cat([next_flat] * self.num_agents, dim=-1)
            centralized_obs = centralized_obs.to(self.device)
            centralized_next_obs = centralized_next_obs.to(self.device)
        else:
            centralized_next_obs = None

        # Potential Game
        current_v = self.critic.get_values(centralized_obs, rnn_states_batch, masks_batch)[0].detach()
        if centralized_next_obs is not None:
            next_v = self.critic.get_values(centralized_next_obs, rnn_states_batch, masks_batch)[0].detach()
        else:
            next_v = torch.zeros_like(current_v)

        delta_v = next_v - current_v.detach()
        denom = current_v.detach().abs().mean(dim=1, keepdim=True).clamp(min=0.1) + 1e-5
        w_p = self.potential_weight * (1 + torch.sigmoid(current_v.detach().mean()/10.0))
        f_p = w_p * torch.tanh(delta_v / denom).detach()
        adv_targ += f_p

        # diff reward
        diff_rewards = []
        for i in range(self.num_agents):
            m = obs_flat.clone()
            start = i * local_dim
            m[:, start:start+local_dim] = 0
            mc = self.critic.get_values(torch.cat([m]*self.num_agents, dim=-1), rnn_states_batch, masks_batch)[0]
            diff_rewards.append(current_v - mc)
        diff_avg = torch.mean(torch.stack(diff_rewards, dim=0), dim=0)
        adv_targ += self.coop_bonus_weight * diff_avg

        # RND
        with torch.no_grad():
            tgt_feat = self.rnd_target(centralized_obs.detach())
        pred_feat = self.rnd_predictor(centralized_obs.detach())
        rnd_err = (tgt_feat - pred_feat).pow(2).sum(dim=1, keepdim=True)
        adv_targ += self.int_rnd_weight * rnd_err.detach()
        loss_rnd = rnd_err.mean()
        self.rnd_optimizer.zero_grad()
        loss_rnd.backward()
        self.rnd_optimizer.step()
        self._rnd_err_queue.append(rnd_err.mean().item())

        action_log_probs, dist_entropy, _ = self.evaluate_actions(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )

        imp_weights = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1,
            keepdim=True,
        )
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self.use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(
                factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True
            ).mean()

        policy_loss = policy_action_loss

        self.actor_optimizer.zero_grad()
        (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())

        self.actor_optimizer.step()

        return policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, actor_buffer, advantages, state_type):
        train_info = {
            "policy_loss": 0,
            "dist_entropy": 0,
            "actor_grad_norm": 0,
            "ratio": 0
        }

        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return train_info

        if state_type == "EP":
            advantages_copy = advantages.copy()
            advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)

            obs = torch.tensor(actor_buffer.obs[:, :-1], dtype=torch.float32, device=self.device)
            next_obs = torch.tensor(actor_buffer.obs[:, 1:], dtype=torch.float32, device=self.device)
            rnn_states_batch = torch.tensor(actor_buffer.rnn_states[:, :-1], dtype=torch.float32, device=self.device)
            masks_batch = torch.tensor(actor_buffer.masks[:, :-1], dtype=torch.float32, device=self.device)

            if obs.ndim == 4:
                B, T, A, D = obs.shape
                obs_flat = obs.reshape(B * T, -1)
                next_obs_flat = next_obs.reshape(B * T, -1)
                rnn_states_batch = rnn_states_batch.reshape(B * T, -1)
                masks_batch = masks_batch.reshape(B * T, -1)
            elif obs.ndim == 3:
                B, T, D = obs.shape
                obs_flat = obs.reshape(B * T, -1)
                next_obs_flat = next_obs.reshape(B * T, -1)
                rnn_states_batch = rnn_states_batch.reshape(B * T, -1)
                masks_batch = masks_batch.reshape(B * T, -1)
            else:
                raise ValueError("Unexpected observation dimension: {}".format(obs.shape))

            centralized_obs = torch.cat([obs_flat] * self.num_agents, dim=-1)
            centralized_next_obs = torch.cat([next_obs_flat] * self.num_agents, dim=-1)

            current_v = self.critic.get_values(centralized_obs, rnn_states_batch, masks_batch)[0].detach()
            next_v = self.critic.get_values(centralized_next_obs, rnn_states_batch, masks_batch)[0].detach()
            delta_v = next_v - current_v

            # potential 奖励
            advantages_flat = advantages.reshape(-1, 1)
            min_len = min(advantages_flat.shape[0], delta_v.shape[0])
            advantages_flat = advantages_flat[:min_len]
            delta_v_np = delta_v.cpu().numpy()[:min_len]
            shaped_adv = advantages_flat + self.potential_weight * delta_v_np
            advantages = (shaped_adv - mean_advantages) / (std_advantages + 1e-5)
            # 只恢复前 min_len 个元素，剩下的保持原值
            advantages = advantages.reshape(-1, 1)
            pad_len = advantages_copy.size - min_len
            if pad_len > 0:
                # 填充剩余部分为0或nan，保证shape一致
                pad = np.zeros((pad_len, 1), dtype=advantages.dtype)
                advantages = np.concatenate([advantages, pad], axis=0)
            advantages = advantages.reshape(*advantages_copy.shape)

        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = actor_buffer.recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch, self.data_chunk_length
                )
            elif self.use_naive_recurrent_policy:
                data_generator = actor_buffer.naive_recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch
                )
            else:
                data_generator = actor_buffer.feed_forward_generator_actor(
                    advantages, self.actor_num_mini_batch
                )

            for sample in data_generator:
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(sample)

                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch
        for k in train_info:
            train_info[k] /= num_updates

        return train_info
