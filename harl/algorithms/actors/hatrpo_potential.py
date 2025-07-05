"""HATRPO algorithm."""

import numpy as np
import torch
import torch.nn as nn
from harl.utils.envs_tools import check
from harl.utils.trpo_util import (
    flat_grad,
    flat_params,
    conjugate_gradient,
    fisher_vector_product,
    update_model,
    kl_divergence,
)
from harl.algorithms.actors.on_policy_base import OnPolicyBase
from harl.models.policy_models.stochastic_policy import StochasticPolicy
from collections import deque


class HATRPO_Potential(OnPolicyBase):
    def __init__(self, args, obs_space, act_space, num_agents, device=torch.device("cpu")):
        """Initialize HATRPO algorithm.
        Args:
            args: (dict) arguments.
            obs_space: (gym.spaces or list) observation space.
            act_space: (gym.spaces) action space.
            device: (torch.device) device to use for tensor operations.
        """
        assert (
            act_space.__class__.__name__ != "MultiDiscrete"
        ), "only continuous and discrete action space is supported by HATRPO."
        super(HATRPO_Potential, self).__init__(args, obs_space, act_space, device)

        self.kl_threshold = args["kl_threshold"]
        self.ls_step = args["ls_step"]
        self.accept_ratio = args["accept_ratio"]
        self.backtrack_coeff = args["backtrack_coeff"]
        
        # Potential Game 相关参数
        self.potential_weight = args.get("potential_weight", 0.25)  # 潜在游戏权重
        self.coop_bonus_weight = args.get("coop_bonus_weight", 0.15)  # 合作奖励权重
        self.int_rnd_weight = args.get("int_rnd_weight", 0.1)  # RND内在奖励权重
        self.num_agents = num_agents

        # RND
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
        """Update actor networks.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
        Returns:
            kl: (torch.Tensor) KL divergence between old and new policy.
            loss_improve: (np.float32) loss improvement.
            expected_improve: (np.ndarray) expected loss improvement.
            dist_entropy: (torch.Tensor) action entropies.
            ratio: (torch.Tensor) ratio between new and old policy.
        """

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

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        factor_batch = check(factor_batch).to(**self.tpdv)

        # 处理不同维度的观察值
        if obs_batch.ndim == 4:  # 4D: (batch_size, num_agents, obs_dim, 1)
            obs_flat = obs_batch.reshape(-1, obs_batch.shape[-2])
            next_obs_flat = obs_batch[:, 1:].reshape(-1, obs_batch.shape[-2])  # 使用当前观察值的下一个时间步
            rnn_states_batch_flat = rnn_states_batch.reshape(-1, *rnn_states_batch.shape[2:])
            masks_batch_flat = masks_batch.reshape(-1, *masks_batch.shape[2:])
        elif obs_batch.ndim == 3:  # 3D: (batch_size, num_agents, obs_dim)
            obs_flat = obs_batch.reshape(-1, obs_batch.shape[-1])
            next_obs_flat = obs_batch[:, 1:].reshape(-1, obs_batch.shape[-1])  # 使用当前观察值的下一个时间步
            rnn_states_batch_flat = rnn_states_batch.reshape(-1, *rnn_states_batch.shape[2:])
            masks_batch_flat = masks_batch.reshape(-1, *masks_batch.shape[2:])
        else:  # 2D: (batch_size, obs_dim)
            obs_flat = obs_batch
            next_obs_flat = obs_batch  # 对于2D情况，使用相同的观察值
            rnn_states_batch_flat = rnn_states_batch.reshape(-1, *rnn_states_batch.shape[1:])
            masks_batch_flat = masks_batch.reshape(-1, *masks_batch.shape[1:])

        # 确保所有输入都是 PyTorch 张量
        obs_flat = torch.tensor(obs_flat, dtype=torch.float32, device=self.device)
        next_obs_flat = torch.tensor(next_obs_flat, dtype=torch.float32, device=self.device)
        rnn_states_batch_flat = torch.tensor(rnn_states_batch_flat, dtype=torch.float32, device=self.device)
        masks_batch_flat = torch.tensor(masks_batch_flat, dtype=torch.float32, device=self.device)

        # 计算当前状态和下一状态的值
        centralized_obs = torch.cat([obs_flat] * self.num_agents, dim=-1)
        centralized_next_obs = torch.cat([next_obs_flat] * self.num_agents, dim=-1)

        current_v = self.critic.get_values(centralized_obs, rnn_states_batch_flat, masks_batch_flat)[0].detach()
        next_v = self.critic.get_values(centralized_next_obs, rnn_states_batch_flat, masks_batch_flat)[0].detach()
        delta_v = next_v - current_v

        # potential
        adv_targ_flat = adv_targ.reshape(-1, 1)
        min_len = min(adv_targ_flat.shape[0], delta_v.shape[0])
        adv_targ_flat = adv_targ_flat[:min_len]
        delta_v = delta_v[:min_len]
        shaped_adv = adv_targ_flat + self.potential_weight * delta_v

        # diff reward
        local_dim = obs_flat.shape[-1] // self.num_agents
        diff_rewards = []
        for i in range(self.num_agents):
            m = obs_flat.clone()
            start = i * local_dim
            m[:, start:start+local_dim] = 0
            mc = self.critic.get_values(torch.cat([m]*self.num_agents, dim=-1), rnn_states_batch_flat, masks_batch_flat)[0]
            diff_rewards.append(current_v - mc)
        diff_avg = torch.mean(torch.stack(diff_rewards, dim=0), dim=0)
        shaped_adv = shaped_adv + self.coop_bonus_weight * diff_avg

        # RND
        with torch.no_grad():
            tgt_feat = self.rnd_target(centralized_obs.detach())
        pred_feat = self.rnd_predictor(centralized_obs.detach())
        rnd_err = (tgt_feat - pred_feat).pow(2).sum(dim=1, keepdim=True)
        shaped_adv = shaped_adv + self.int_rnd_weight * rnd_err.detach()
        loss_rnd = rnd_err.mean()
        self.rnd_optimizer.zero_grad()
        loss_rnd.backward()
        self.rnd_optimizer.step()

        adv_targ = shaped_adv.reshape(*adv_targ.shape)

        # Reshape to do evaluations for all steps in a single forward pass
        action_log_probs, dist_entropy, _ = self.evaluate_actions(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )

        # actor update
        ratio = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1,
            keepdim=True,
        )
        if self.use_policy_active_masks:
            loss = (
                torch.sum(ratio * factor_batch * adv_targ, dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            loss = torch.sum(
                ratio * factor_batch * adv_targ, dim=-1, keepdim=True
            ).mean()

        loss_grad = torch.autograd.grad(
            loss, self.actor.parameters(), allow_unused=True
        )
        loss_grad = flat_grad(loss_grad)

        step_dir = conjugate_gradient(
            self.actor,
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
            loss_grad.data,
            nsteps=10,
            device=self.device,
        )

        loss = loss.data.cpu().numpy()

        params = flat_params(self.actor)
        fvp = fisher_vector_product(
            self.actor,
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
            step_dir,
        )
        shs = 0.5 * (step_dir * fvp).sum(0, keepdim=True)
        step_size = 1 / torch.sqrt(shs / self.kl_threshold)[0]
        full_step = step_size * step_dir

        old_actor = StochasticPolicy(
            self.args, self.obs_space, self.act_space, self.device
        )
        update_model(old_actor, params)
        expected_improve = (loss_grad * full_step).sum(0, keepdim=True)
        expected_improve = expected_improve.data.cpu().numpy()

        # Backtracking line search (https://en.wikipedia.org/wiki/Backtracking_line_search)
        flag = False
        fraction = 1
        for i in range(self.ls_step):
            new_params = params + fraction * full_step
            update_model(self.actor, new_params)
            action_log_probs, dist_entropy, _ = self.evaluate_actions(
                obs_batch,
                rnn_states_batch,
                actions_batch,
                masks_batch,
                available_actions_batch,
                active_masks_batch,
            )

            ratio = getattr(torch, self.action_aggregation)(
                torch.exp(action_log_probs - old_action_log_probs_batch),
                dim=-1,
                keepdim=True,
            )
            if self.use_policy_active_masks:
                new_loss = (
                    torch.sum(ratio * factor_batch * adv_targ, dim=-1, keepdim=True)
                    * active_masks_batch
                ).sum() / active_masks_batch.sum()
            else:
                new_loss = torch.sum(
                    ratio * factor_batch * adv_targ, dim=-1, keepdim=True
                ).mean()

            new_loss = new_loss.data.cpu().numpy()
            loss_improve = new_loss - loss

            kl = kl_divergence(
                obs_batch,
                rnn_states_batch,
                actions_batch,
                masks_batch,
                available_actions_batch,
                active_masks_batch,
                new_actor=self.actor,
                old_actor=old_actor,
            )
            kl = kl.mean()

            if (
                kl < self.kl_threshold
                and (loss_improve / expected_improve) > self.accept_ratio
                and loss_improve.item() > 0
            ):
                flag = True
                break
            expected_improve *= self.backtrack_coeff
            fraction *= self.backtrack_coeff

        if not flag:
            params = flat_params(old_actor)
            update_model(self.actor, params)
            print("policy update does not impove the surrogate")

        return kl, loss_improve, expected_improve, dist_entropy, ratio

    def train(self, actor_buffer, advantages, state_type):
        """Perform a training update using minibatch GD.
        Args:
            actor_buffer: (OnPolicyActorBuffer) buffer containing training data related to actor.
            advantages: (np.ndarray) advantages.
            state_type: (str) type of state.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info["kl"] = 0
        train_info["dist_entropy"] = 0
        train_info["loss_improve"] = 0
        train_info["expected_improve"] = 0
        train_info["ratio"] = 0

        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return train_info

        if state_type == "EP":
            advantages_copy = advantages.copy()
            advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)

            # 修改 Advantage 计算，加入 Potential Game 影响
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
            elif obs.ndim == 2:
                B, D = obs.shape
                obs_flat = obs
                next_obs_flat = obs
                rnn_states_batch = rnn_states_batch.reshape(B, -1)
                masks_batch = masks_batch.reshape(B, -1)
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

        if self.use_recurrent_policy:
            data_generator = actor_buffer.recurrent_generator_actor(
                advantages, 1, self.data_chunk_length
            )
        elif self.use_naive_recurrent_policy:
            data_generator = actor_buffer.naive_recurrent_generator_actor(advantages, 1)
        else:
            data_generator = actor_buffer.feed_forward_generator_actor(advantages, 1)

        for sample in data_generator:
            kl, loss_improve, expected_improve, dist_entropy, imp_weights = self.update(
                sample
            )

            train_info["kl"] += kl
            train_info["loss_improve"] += loss_improve.item()
            train_info["expected_improve"] += expected_improve
            train_info["dist_entropy"] += dist_entropy.item()
            train_info["ratio"] += imp_weights.mean()

        num_updates = 1

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info
