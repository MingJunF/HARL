"""MAPPO_potential algorithm."""
import numpy as np
import torch
import torch.nn as nn
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm
from harl.algorithms.actors.on_policy_base import OnPolicyBase


class MAPPO_Potential(OnPolicyBase):
    def __init__(self, args, obs_space, act_space, num_agents, device=torch.device("cpu")):
        super(MAPPO_Potential, self).__init__(args, obs_space, act_space, device)

        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]

        self.potential_weight = args.get("potential_weight", 0.3) # default 0.1
        self.exploration_bonus_weight = args.get("exploration_bonus_weight", 0.02) #default 0.02
        self.stagnation_penalty = args.get("stagnation_penalty", 0.1) #default 0.1
        self.potential_stagnation_eps = args.get("potential_stagnation_eps", 1e-3)
        self.num_agents = num_agents

        self._stagnation_count = None
        self._entropy_adaptor = 0.5

    def update(self, sample):

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
        ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        obs_batch = check(obs_batch).to(**self.tpdv)
        next_obs_batch = check(next_obs_batch).to(**self.tpdv) if next_obs_batch is not None else None

        if next_obs_batch is not None and next_obs_batch.ndim == 3:
            next_obs_batch = next_obs_batch[:, 0, :]

        batch_size = obs_batch.shape[0]
        agent_num = self.num_agents
        obs_batch_flat = obs_batch.reshape(batch_size, -1)
        centralized_obs = torch.cat([obs_batch_flat] * agent_num, dim=-1)

        current_v, _ = self.critic.critic(centralized_obs, rnn_states_batch, masks_batch)

        if not hasattr(self, '_stagnation_count'):
            self._stagnation_count = torch.zeros_like(current_v, device=self.device)
        elif (self._stagnation_count is None or
              self._stagnation_count.shape[0] != batch_size or
              self._stagnation_count.device != current_v.device):
            self._stagnation_count = torch.zeros_like(current_v, device=self.device)

        with torch.no_grad():
            if next_obs_batch is not None:
                next_obs_batch_flat = next_obs_batch.reshape(batch_size, -1)
                centralized_next_obs = torch.cat([next_obs_batch_flat] * agent_num, dim=-1)
                next_v, _ = self.critic.critic(centralized_next_obs, rnn_states_batch, masks_batch)
                next_v = next_v.detach()
            else:
                next_v = torch.zeros_like(current_v)

        with torch.no_grad():
            delta_v = next_v - current_v.detach()
            adaptive_weight = self.potential_weight * (1 + torch.sigmoid(current_v.detach().mean() / 10.0))
            delta_v_normalized = delta_v / (current_v.detach().abs().mean(dim=1, keepdim=True).clamp(min=1.0) + 1e-5)
            potential_term = adaptive_weight * torch.tanh(delta_v_normalized)

        adv_targ = adv_targ + potential_term

        action_log_probs, dist_entropy, _ = self.evaluate_actions(
            obs_batch, rnn_states_batch, actions_batch,
            masks_batch, available_actions_batch, active_masks_batch
        )

        with torch.no_grad():
            entropy_ratio = (dist_entropy.detach() / self.entropy_coef).clamp(0.1, 10.0)
            dynamic_threshold = self.potential_stagnation_eps * (0.5 + 0.5 * torch.sigmoid(entropy_ratio - 1.0))
            stagnation_mask = (delta_v.abs() < dynamic_threshold).float().to(self.device)

            self._stagnation_count = self._stagnation_count.to(self.device)
            self._stagnation_count = 0.9 * self._stagnation_count + 0.1 * stagnation_mask
            adv_targ = adv_targ - self.stagnation_penalty * (self._stagnation_count > 0.5).float()

        exploration_bonus = 0.0
        if self.exploration_bonus_weight > 0:
            with torch.enable_grad():
                centralized_obs.requires_grad_(True)
                current_v_for_grad, _ = self.critic.critic(centralized_obs, rnn_states_batch, masks_batch)
                grad = torch.autograd.grad(current_v_for_grad.sum(), centralized_obs, retain_graph=True)[0]
                centralized_obs.requires_grad_(False)

                with torch.no_grad():
                    noise = torch.randn_like(centralized_obs) * torch.exp(-grad.abs().clamp(max=10.0))
                    q_deviated, _ = self.critic.critic(centralized_obs + noise, rnn_states_batch, masks_batch)
                    exploration_bonus = (q_deviated - current_v.detach()).clamp(min=0).mean()
        imp_weights = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1,
            keepdim=True,
        )
        surr1 = imp_weights * adv_targ.detach()
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ.detach()

        if self.use_policy_active_masks:
            policy_action_loss = (-torch.min(surr1, surr2) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.min(surr1, surr2).mean()

        policy_loss = policy_action_loss - self.entropy_coef * dist_entropy
        if self.exploration_bonus_weight > 0:
            policy_loss = policy_loss - self.exploration_bonus_weight * exploration_bonus

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
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

            # 修改 Advantage 计算，加入 Potential Game 影响
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
            normalized_advantages = (advantages_ori_tensor - mean_advantages) / (
                std_advantages + 1e-5
            )
            advantages_list = []
            for agent_id in range(num_agents):
                advantages_list.append(normalized_advantages[agent_id])
        elif state_type == "FP":
            advantages_list = []
            for agent_id in range(num_agents):
                advantages[:, :, agent_id] = advantages[:, :, agent_id] + self.potential_weight * (
                        self.critic.v_net(actor_buffer[agent_id].obs) - self.critic.v_net(
                    actor_buffer[agent_id].obs).detach()
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
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(
                    tuple(batches)
                )

                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info
