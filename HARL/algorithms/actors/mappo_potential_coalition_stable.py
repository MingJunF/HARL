"""
MAPPO_Potential algorithm with:
- Potential-based shaping
- Cooperation difference reward
- Goal-Centric Coalition Consistency bonus (GCCC)
"""

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

        self.potential_weight = args.get("potential_weight", 0.25)
        self.coop_bonus_weight = args.get("coop_bonus_weight", 0.15)
        self.coalition_bonus_weight = args.get("coalition_bonus_weight", 0.2)

        self.num_agents = num_agents
        self.prev_goal_agent_visibility = None

    def compute_core_stability_bonus(self, goal_agent_visibility):
        if self.prev_goal_agent_visibility is None:
            self.prev_goal_agent_visibility = torch.zeros_like(goal_agent_visibility)

        coalition_mask_t = goal_agent_visibility.float()
        coalition_mask_prev = self.prev_goal_agent_visibility.float()

        same_membership = coalition_mask_t * coalition_mask_prev
        stability_score = same_membership.sum(dim=-1) / (coalition_mask_t.sum(dim=-1) + 1e-5)
        core_bonus = (coalition_mask_t * stability_score.unsqueeze(-1)).sum(dim=1)

        self.prev_goal_agent_visibility = goal_agent_visibility.detach()
        return core_bonus.view(-1, 1)

    def update(self, sample):
        (obs_batch, next_obs_batch, rnn_states_batch, actions_batch,
         masks_batch, active_masks_batch, old_action_log_probs_batch,
         adv_targ, available_actions_batch, goal_agent_visibility) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        obs_batch = check(obs_batch).to(**self.tpdv)
        next_obs_batch = check(next_obs_batch).to(**self.tpdv) if next_obs_batch is not None else None
        if goal_agent_visibility is not None and isinstance(goal_agent_visibility, torch.Tensor):
            goal_agent_visibility = check(goal_agent_visibility).to(**self.tpdv)
        else:
            goal_agent_visibility = None

        if next_obs_batch is not None and next_obs_batch.ndim == 3:
            next_obs_batch = next_obs_batch[:, 0, :]

        batch_size = obs_batch.shape[0]
        agent_num = self.num_agents
        obs_batch_flat = obs_batch.reshape(batch_size, -1)

        centralized_obs = torch.cat([obs_batch_flat] * agent_num, dim=-1)
        current_v, _ = self.critic.critic(centralized_obs, rnn_states_batch, masks_batch)
        if next_obs_batch is not None:
            next_obs_flat = next_obs_batch.reshape(batch_size, -1)
            centralized_next_obs = torch.cat([next_obs_flat] * agent_num, dim=-1)
            next_v, _ = self.critic.critic(centralized_next_obs, rnn_states_batch, masks_batch)
            next_v = next_v.detach()
        else:
            next_v = torch.zeros_like(current_v)

        # Potential-based shaping
        delta_v = next_v - current_v.detach()
        denom = current_v.detach().abs().mean(dim=1, keepdim=True).clamp(min=0.1) + 1e-5
        adaptive_weight = self.potential_weight * (1 + torch.sigmoid(current_v.detach().mean() / 10.0))
        potential_term = adaptive_weight * torch.tanh(delta_v / denom).detach()
        adv_targ = adv_targ + potential_term

        # Cooperation difference reward
        diff_rewards = []
        local_dim = obs_batch.shape[1] // agent_num
        for i in range(agent_num):
            masked_obs = obs_batch_flat.clone()
            masked_obs[:, i * local_dim:(i + 1) * local_dim] = 0
            masked_centralized_obs = torch.cat([masked_obs] * agent_num, dim=-1)
            baseline_i, _ = self.critic.critic(masked_centralized_obs, rnn_states_batch, masks_batch)
            diff_rewards.append(current_v - baseline_i)
        diff_reward_avg = torch.mean(torch.stack(diff_rewards, dim=0), dim=0)
        adv_targ = adv_targ + self.coop_bonus_weight * diff_reward_avg

        # GCCC: Goal-centric coalition consistency
        if goal_agent_visibility is not None:
            gccc_bonus = self.compute_core_stability_bonus(goal_agent_visibility)
            gccc_bonus = gccc_bonus / (gccc_bonus.max() + 1e-5)
            adv_targ = adv_targ + self.coalition_bonus_weight * gccc_bonus.detach()

        action_log_probs, dist_entropy, _ = self.evaluate_actions(
            obs_batch, rnn_states_batch, actions_batch,
            masks_batch, available_actions_batch, active_masks_batch
        )
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = imp_weights * adv_targ.detach()
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ.detach()
        if self.use_policy_active_masks:
            policy_action_loss = (-torch.min(surr1, surr2) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.min(surr1, surr2).mean()

        policy_loss = policy_action_loss - self.entropy_coef * dist_entropy

        self.actor_optimizer.zero_grad()
        policy_loss.backward(retain_graph=False)
        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())

        self.actor_optimizer.step()

        return policy_loss.detach(), dist_entropy.detach(), actor_grad_norm, imp_weights.detach()

    def train(self, actor_buffer, advantages, state_type):
        train_info = {"policy_loss": 0, "dist_entropy": 0, "actor_grad_norm": 0, "ratio": 0}
        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return train_info

        if state_type == "EP":
            advantages_copy = advantages.copy()
            advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            current_v = self.critic.v_net(actor_buffer.obs).detach()
            next_v = self.critic.v_net(actor_buffer.obs).detach()
            advantages = advantages + self.potential_weight * (next_v - current_v)
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
                if len(sample) == 9:
                    sample = list(sample) + [None]  # add goal_agent_visibility placeholder
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(tuple(sample))
                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates
        return train_info

    def share_param_train(self, actor_buffer, advantages, num_agents, state_type):
        train_info = {"policy_loss": 0, "dist_entropy": 0, "actor_grad_norm": 0, "ratio": 0}
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
            advantages_list = [normalized_advantages[agent_id] for agent_id in range(num_agents)]
        elif state_type == "FP":
            advantages_list = []
            for agent_id in range(num_agents):
                advantages[:, :, agent_id] = advantages[:, :, agent_id] + self.potential_weight * (
                        self.critic.v_net(actor_buffer[agent_id].obs) - self.critic.v_net(actor_buffer[agent_id].obs).detach())
                advantages_list.append(advantages[:, :, agent_id])

        for _ in range(self.ppo_epoch):
            data_generators = []
            for agent_id in range(num_agents):
                if self.use_recurrent_policy:
                    data_generator = actor_buffer[agent_id].recurrent_generator_actor(
                        advantages_list[agent_id], self.actor_num_mini_batch, self.data_chunk_length)
                elif self.use_naive_recurrent_policy:
                    data_generator = actor_buffer[agent_id].naive_recurrent_generator_actor(
                        advantages_list[agent_id], self.actor_num_mini_batch)
                else:
                    data_generator = actor_buffer[agent_id].feed_forward_generator_actor(
                        advantages_list[agent_id], self.actor_num_mini_batch)
                data_generators.append(data_generator)

            for _ in range(self.actor_num_mini_batch):
                batches = [[] for _ in range(10)]
                for generator in data_generators:
                    sample = next(generator)
                    for i in range(9):
                        batches[i].append(sample[i])
                    batches[9].append(None)  # placeholder for goal_agent_visibility
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
