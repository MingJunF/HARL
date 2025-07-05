"""MAPPO_potential algorithm with intrinsic RND exploration bonus."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm
from harl.algorithms.actors.on_policy_base import OnPolicyBase


class MAPPO_Potential(OnPolicyBase):
    def __init__(self, args, obs_space, act_space, num_agents, device=torch.device("cpu")):
        super(MAPPO_Potential, self).__init__(args, obs_space, act_space, device)

        # PPO 超参
        self.clip_param           = args["clip_param"]
        self.ppo_epoch            = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef         = args["entropy_coef"]
        self.use_max_grad_norm    = args["use_max_grad_norm"]
        self.max_grad_norm        = args["max_grad_norm"]

        # 分区潜在塑形超参
        self.num_agents        = num_agents
        self.potential_weight  = args.get("potential_weight", 0.3)
        self.gamma             = args.get("gamma", 0.99)

        # 差分合作奖励
        self.coop_bonus_weight = args.get("coop_bonus_weight", 0.3)

        # 局部观测与全局拼接维度
        local_dim      = int(np.prod(obs_space.shape))
        global_dim     = local_dim * num_agents

        # 每个 agent 的局部 RND 网络
        self.int_rnd_weight = args.get("int_rnd_weight", 0.05)
        self.local_rnd_targets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(local_dim, 64), nn.ReLU(),
                nn.Linear(64, 64)
            ) for _ in range(num_agents)
        ])
        for net in self.local_rnd_targets:
            for p in net.parameters(): p.requires_grad = False
        self.local_rnd_preds = nn.ModuleList([
            nn.Sequential(
                nn.Linear(local_dim, 64), nn.ReLU(),
                nn.Linear(64, 64)
            ) for _ in range(num_agents)
        ])
        self.local_rnd_opts = [
            torch.optim.Adam(net.parameters(), lr=1e-4)
            for net in self.local_rnd_preds
        ]

        # 分区潜在函数 Phi 网络及优化器
        self.phi_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(local_dim, 64), nn.ReLU(),
                nn.Linear(64, 1)
            ) for _ in range(num_agents)
        ])
        self.phi_opts = [
            torch.optim.Adam(net.parameters(), lr=1e-3)
            for net in self.phi_nets
        ]

    def update(self, sample):
        # 解包
        (obs_batch, next_obs_batch, rnn_states_batch, actions_batch,
         masks_batch, active_masks_batch, old_action_log_probs_batch,
         adv_targ, available_actions_batch) = sample

        # 转 device
        obs = check(obs_batch).to(**self.tpdv)
        nxt = check(next_obs_batch).to(**self.tpdv) if next_obs_batch is not None else None
        old_logp = check(old_action_log_probs_batch).to(**self.tpdv)
        adv = check(adv_targ).to(**self.tpdv)
        amask = check(active_masks_batch).to(**self.tpdv)

        if nxt is not None and nxt.ndim == 3:
            nxt = nxt[:, 0, :]

        B, local_dim = obs.shape
        N            = self.num_agents

        # 构造全局 obs
        obs_flat        = obs.view(B, -1)
        centralized_obs = obs_flat.repeat(1, N)
        if nxt is not None:
            nxt_flat        = nxt.view(B, -1)
            centralized_nxt = nxt_flat.repeat(1, N)
        else:
            centralized_nxt = None

        # --- 1) 分区潜在塑形 ---
        obs_parts = obs.view(1, B, local_dim).repeat(N, 1, 1)
        nxt_parts = nxt.view(1, B, local_dim).repeat(N, 1, 1) if nxt is not None else None
        shape_sum = torch.zeros((B, 1), device=adv.device)
        for i, phi in enumerate(self.phi_nets):
            s_i    = obs_parts[i]             # [B, local_dim]
            phi_s  = phi(s_i)                 # [B, 1]
            if nxt_parts is not None:
                phi_sp = phi(nxt_parts[i].detach())
            else:
                phi_sp = torch.zeros_like(phi_s)
            f_i    = self.potential_weight * (self.gamma * phi_sp - phi_s)
            shape_sum += f_i
            # 可选：训练 phi 网络去拟合局部 critic
            # v_i, _ = self.critic.critic(torch.cat([s_i]*N, dim=-1), rnn_states_batch, masks_batch)
            # loss_phi = F.mse_loss(phi_s, v_i.detach())
            # self.phi_opts[i].zero_grad(); loss_phi.backward(); self.phi_opts[i].step()
        adv = adv + shape_sum

        # --- 2) 差分合作奖励 ---
        # 先计算全局价值
        current_v, _ = self.critic.critic(centralized_obs, rnn_states_batch, masks_batch)
        diff_rewards = []
        for i in range(N):
            m   = obs_flat.clone()
            start = i * local_dim
            m[:, start:start+local_dim] = 0
            mc, _ = self.critic.critic(m.repeat(1, N), rnn_states_batch, masks_batch)
            diff_rewards.append(current_v - mc)
        diff_avg = torch.mean(torch.stack(diff_rewards, dim=0), dim=0)
        adv = adv + self.coop_bonus_weight * diff_avg

        # --- 3) 局部 RND 探索奖励 ---
        rnd_sum = torch.zeros((B, 1), device=adv.device)
        obs_parts = obs_parts  # reuse for local parts
        for i in range(N):
            s_i = obs_parts[i]             # [B, local_dim]
            with torch.no_grad():
                tgt = self.local_rnd_targets[i](s_i)
            pred = self.local_rnd_preds[i](s_i)
            err  = (tgt - pred).pow(2).sum(dim=1, keepdim=True)
            rnd_sum += err
            # 更新 predictor
            loss_rnd = err.mean()
            self.local_rnd_opts[i].zero_grad()
            loss_rnd.backward()
            self.local_rnd_opts[i].step()
        adv = adv + (self.int_rnd_weight / N) * rnd_sum

        # --- 4) PPO 损失 & 更新 ---
        action_logp, dist_entropy, _ = self.evaluate_actions(
            obs, rnn_states_batch, actions_batch,
            masks_batch, available_actions_batch, amask
        )
        imp_weights = torch.exp(action_logp - old_logp)
        s1 = imp_weights * adv.detach()
        s2 = torch.clamp(imp_weights, 1 - self.clip_param, 1 + self.clip_param) * adv.detach()
        if self.use_policy_active_masks:
            pa_loss = (-torch.min(s1, s2) * amask).sum() / amask.sum()
        else:
            pa_loss = -torch.min(s1, s2).mean()
        policy_loss = pa_loss - self.entropy_coef * dist_entropy

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())
        self.actor_optimizer.step()

        return policy_loss.detach(), dist_entropy.detach(), actor_grad_norm, imp_weights.detach()

    def train(self, actor_buffer, advantages, state_type):
        """Perform a training update for non-parameter-sharing MAPPO using minibatch GD.
        Args:
            actor_buffer: (OnPolicyActorBuffer) buffer containing training data related to actor.
            advantages: (np.ndarray) advantages.
            state_type: (str) type of state.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0

        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return train_info

        if state_type == "EP":
            advantages_copy = advantages.copy()
            advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)

            # 修改 Advantage 计算，加入 Potential Game 影响-jianglin
            current_v = self.critic.v_net(actor_buffer.obs).detach()
            next_v = self.critic.v_net(actor_buffer.obs).detach()
            advantages = advantages + self.potential_weight * (next_v - current_v)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

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

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def share_param_train(self, actor_buffer, advantages, num_agents, state_type):
        """Perform a training update for parameter-sharing MAPPO using minibatch GD.
        Args:
            actor_buffer: (list[OnPolicyActorBuffer]) buffer containing training data related to actor.
            advantages: (np.ndarray) advantages.
            num_agents: (int) number of agents.
            state_type: (str) type of state.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0

        if state_type == "EP":
            advantages_ori_list = []
            advantages_copy_list = []
            for agent_id in range(num_agents):
                advantages_ori = advantages.copy()
                advantages_ori_list.append(advantages_ori)
                advantages_copy = advantages.copy()
                advantages_copy[actor_buffer[agent_id].active_masks[:-1] == 0.0] = np.nan
                advantages_copy_list.append(advantages_copy)
            advantages_ori_tensor = np.array(advantages_ori_list)
            advantages_copy_tensor = np.array(advantages_copy_list)
            mean_advantages = np.nanmean(advantages_copy_tensor)
            std_advantages = np.nanstd(advantages_copy_tensor)
            normalized_advantages = (advantages_ori_tensor - mean_advantages) / (std_advantages + 1e-5)
            advantages_list = []
            for agent_id in range(num_agents):
                advantages_list.append(normalized_advantages[agent_id])
        elif state_type == "FP":
            advantages_list = []
            for agent_id in range(num_agents):
                advantages[:, :, agent_id] = advantages[:, :, agent_id] + self.potential_weight * (
                        self.critic.v_net(actor_buffer[agent_id].obs) - self.critic.v_net(actor_buffer[agent_id].obs).detach()
                )
                advantages_list.append(advantages[:, :, agent_id])

        for _ in range(self.ppo_epoch):
            data_generators = []
            for agent_id in range(num_agents):
                if self.use_recurrent_policy:
                    data_generator = actor_buffer[agent_id].recurrent_generator_actor(
                        advantages_list[agent_id],
                        self.actor_num_mini_batch,
                        self.data_chunk_length,
                    )
                elif self.use_naive_recurrent_policy:
                    data_generator = actor_buffer[agent_id].naive_recurrent_generator_actor(
                        advantages_list[agent_id], self.actor_num_mini_batch
                    )
                else:
                    data_generator = actor_buffer[agent_id].feed_forward_generator_actor(
                        advantages_list[agent_id], self.actor_num_mini_batch
                    )
                data_generators.append(data_generator)

            for _ in range(self.actor_num_mini_batch):
                batches = [[] for _ in range(9)]
                for generator in data_generators:
                    sample = next(generator)
                    for i in range(9):
                        batches[i].append(sample[i])
                for i in range(8):
                    batches[i] = np.concatenate(batches[i], axis=0)
                if batches[8][0] is None:
                    batches[8] = None
                else:
                    batches[8] = np.concatenate(batches[8], axis=0)
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(tuple(batches))
                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates
        return train_info
