import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # 用于 F.mse_loss
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm
from harl.algorithms.actors.on_policy_base import OnPolicyBase

# 示例的 RewardNetwork（请根据任务需要进行修改）
class RewardNetwork(nn.Module):
    def __init__(self, args):
        super(RewardNetwork, self).__init__()
        # 例如，输入维度从 args 获取，默认 64
        input_dim = int(args.get("reward_obs_dim", 64))
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, obs, rnn_reward, masks, act_onehot):
        # 简单示例：仅使用 obs 计算奖励
        return self.fc(obs), None

class R_MAPPO(OnPolicyBase):
    """Reward-enhanced MAPPO algorithm."""
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """
        Initialize R_MAPPO.
        Args:
            args: (dict) 参数字典。
            obs_space: 观测空间。
            act_space: 动作空间。
            device: 使用的设备（cpu/gpu）。
        """
        super(R_MAPPO, self).__init__(args, obs_space, act_space, device)
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]
        # num_rew 为奖励网络个数，转换为整数（清理逗号）
        self.num_rew = int(args.get("num_rew", "3").replace(",", "").strip())
        # 初始化奖励网络及其优化器
        self.rewards = [RewardNetwork(args) for _ in range(self.num_rew)]
        self.reward_optimizers = [
            torch.optim.Adam(self.rewards[i].parameters(), lr=float(args.get("reward_lr", 0.001)))
            for i in range(self.num_rew)
        ]

    def update(self, sample, update_actor=True):
        """
        更新 actor 网络，并调用 reward_update() 更新奖励网络。
        sample 的预期结构（共 12 个元素）：
            0: obs_batch
            1: rnn_states_batch
            2: actions_batch
            3: masks_batch
            4: active_masks_batch
            5: old_action_log_probs_batch
            6: adv_targ
            7: available_actions_batch
            8: rewards_batch
            9: reward_pred_batch
            10: rnn_reward_batch
            11: act_onehot_batch
        """
        (obs_batch, rnn_states_batch, actions_batch, masks_batch, active_masks_batch,
         old_action_log_probs_batch, adv_targ, available_actions_batch, rewards_batch,
         reward_pred_batch, rnn_reward_batch, act_onehot_batch) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        actions_batch = check(actions_batch).to(dtype=torch.int64, device=self.device)

        # 调用 evaluate_actions() 计算动作对数概率、熵（该方法在 OnPolicyBase 中实现）
        action_log_probs, dist_entropy, _ = self.evaluate_actions(
            obs_batch, rnn_states_batch, actions_batch, masks_batch,
            available_actions_batch, active_masks_batch)
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self.use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
                                  * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss
        self.actor_optimizer.zero_grad()
        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()
        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())
        self.actor_optimizer.step()

        # 更新奖励网络
        reward_loss = self.reward_update(sample)
        return policy_loss, dist_entropy, actor_grad_norm, imp_weights, reward_loss

    def reward_update(self, sample):
        """
        更新奖励网络（sample 中包含 12 个元素）：
          0: obs_batch
          1: rnn_states_batch
          2: actions_batch
          3: masks_batch
          4: active_masks_batch
          5: old_action_log_probs_batch
          6: adv_targ
          7: available_actions_batch
          8: rewards_batch
          9: reward_pred_batch
          10: rnn_reward_batch
          11: act_onehot_batch
        """
        (obs_batch, rnn_states_batch, actions_batch, masks_batch, active_masks_batch,
         old_action_log_probs_batch, adv_targ, available_actions_batch, rewards_batch,
         reward_pred_batch, rnn_reward_batch, act_onehot_batch) = sample

        # 如果 rnn_reward_batch 为空，则直接返回0.0（跳过奖励网络更新）
        if isinstance(rnn_reward_batch, np.ndarray) and rnn_reward_batch.size == 0:
            return 0.0

        share_obs_batch = obs_batch  # 使用 obs_batch 作为共享观察

        actions_batch = check(actions_batch).to(dtype=torch.int64, device=self.device)
        rewards_batch = check(rewards_batch).to(**self.tpdv)
        act_onehot_batch = check(act_onehot_batch)
        if not isinstance(act_onehot_batch, torch.Tensor):
            act_onehot_batch = torch.tensor(act_onehot_batch, **self.tpdv)
        else:
            act_onehot_batch = act_onehot_batch.to(**self.tpdv)

        rew_loss = 0
        for i in range(self.num_rew):
            # 调用奖励网络：注意使用 self.rewards[i] 而非 self.policy.rewards[i]
            rewards, _ = self.rewards[i](share_obs_batch, rnn_reward_batch[i], masks_batch, act_onehot_batch)
            reward = torch.gather(rewards, dim=-1, index=actions_batch)
            error = F.mse_loss(rewards_batch - reward)
            current_loss = error.mean()
            self.reward_optimizers[i].zero_grad()
            current_loss.backward()
            _ = nn.utils.clip_grad_norm_(self.rewards[i].parameters(), self.max_grad_norm)
            self.reward_optimizers[i].step()
            rew_loss = current_loss  # 这里只取最后一次的 loss，可根据需要累加
        return rew_loss.item()

    def train(self, actor_buffer, advantages, state_type):
        """
        非参数共享训练：使用 minibatch 梯度下降更新 actor 网络（R_MAPPO）。
        Args:
            actor_buffer: OnPolicyActorBuffer，包含训练数据。
            advantages: np.ndarray 优势值。
            state_type: (str) 状态类型 ("EP" 或 "FP")。
        Returns:
            train_info: dict 包含训练更新信息。
        """
        train_info = {"policy_loss": 0, "dist_entropy": 0, "actor_grad_norm": 0, "ratio": 0, "reward_loss": 0}
        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return train_info

        if state_type == "EP":
            advantages_copy = advantages.copy()
            advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = actor_buffer.recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch, self.data_chunk_length)
            elif self.use_naive_recurrent_policy:
                data_generator = actor_buffer.naive_recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch)
            else:
                data_generator = actor_buffer.feed_forward_generator_actor(
                    advantages, self.actor_num_mini_batch)

            for sample in data_generator:
                policy_loss, dist_entropy, actor_grad_norm, imp_weights, reward_loss = self.update(sample)
                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()
                train_info["reward_loss"] += reward_loss

        num_updates = self.ppo_epoch * self.actor_num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def share_param_train(self, actor_buffer, advantages, num_agents, state_type):
        """
        参数共享训练：对多个 agent 使用共享参数进行更新。
        Args:
            actor_buffer: list[OnPolicyActorBuffer]，每个 agent 对应一个 buffer。
            advantages: np.ndarray 优势值。
            num_agents: (int) agent 数量。
            state_type: (str) 状态类型 ("EP" 或 "FP")。
        Returns:
            train_info: dict 包含训练更新信息。
        """
        train_info = {"policy_loss": 0, "dist_entropy": 0, "actor_grad_norm": 0, "ratio": 0, "reward_loss": 0}
        # 扩展 advantages 至 3 维，并确保第三维为 num_agents
        if advantages.ndim == 2:
            advantages = np.expand_dims(advantages, axis=2)
        if advantages.shape[2] != num_agents:
            advantages = np.tile(advantages, (1, 1, num_agents))
        assert advantages.ndim == 3, f"Expected 3D advantages, got {advantages.shape}"
        #print(f"Shape of advantages before share_param_train: {advantages.shape}")

        if state_type == "EP":
            advantages_copy = advantages.copy()
            for agent_id in range(num_agents):
                mask = actor_buffer[agent_id].active_masks[:-1].squeeze(-1)
                advantages_copy[:, :, agent_id][mask == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            normalized_advantages = (advantages_copy - mean_advantages) / (std_advantages + 1e-5)
            advantages_list = [normalized_advantages[:, :, agent_id] for agent_id in range(num_agents)]
        elif state_type == "FP":
            advantages_list = [advantages[:, :, 0] for _ in range(num_agents)]
        else:
            raise ValueError("Unknown state_type. It must be 'EP' or 'FP'.")

        data_generators = []
        for agent_id in range(num_agents):
            if self.use_recurrent_policy:
                generator = actor_buffer[agent_id].recurrent_generator_actor(
                    advantages_list[agent_id], self.actor_num_mini_batch, self.data_chunk_length)
            elif self.use_naive_recurrent_policy:
                generator = actor_buffer[agent_id].naive_recurrent_generator_actor(
                    advantages_list[agent_id], self.actor_num_mini_batch)
            else:
                generator = actor_buffer[agent_id].feed_forward_generator_actor(
                    advantages_list[agent_id], self.actor_num_mini_batch)
            data_generators.append(generator)

        # 固定预期每个 sample 的元素个数为 12（update() 需要 12 个元素）
        num_elements = 12
        for _ in range(self.ppo_epoch):
            for _ in range(self.actor_num_mini_batch):
                batches = [[] for _ in range(num_elements)]
                for generator in data_generators:
                    try:
                        sample = next(generator)
                    except StopIteration:
                        continue
                    sample = list(sample)
                    if len(sample) < num_elements:
                        sample.extend([np.array([])] * (num_elements - len(sample)))
                    sample = tuple(sample)
                    for i in range(num_elements):
                        batches[i].append(sample[i])
                if any(len(batch) == 0 for batch in batches):
                    continue
                for i in range(num_elements - 1):
                    batches[i] = np.concatenate(batches[i], axis=0)
                if batches[num_elements - 1][0] is None:
                    batches[num_elements - 1] = None
                else:
                    batches[num_elements - 1] = np.concatenate(batches[num_elements - 1], axis=0)
                policy_loss, dist_entropy, actor_grad_norm, imp_weights, reward_loss = self.update(tuple(batches))
                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()
                train_info["reward_loss"] += reward_loss

        num_updates = self.ppo_epoch * self.actor_num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info
