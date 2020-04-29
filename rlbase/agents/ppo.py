import numpy as np
import torch
import torch.nn as nn

from .base import BaseAgent
from policies.actor_critic import ActorCritic
from core.replay_buffer import Memory
from utils.block_entropy import get_blocks, sample_blocks, block_entropy


"""
Advantage Actor-Critic Proximal Policy Optimization
"""


class PPO(BaseAgent):
    def __init__(self, config):
        super(PPO, self).__init__(config)

        self.set_network_configs()

        self.policy = ActorCritic(config).to(self.device)

        self.optimizer = config.training.optim(
            self.policy.parameters(), lr=self.config.training.lr
        )

        self.lr_scheduler = self.config.training.lr_scheduler(
            self.optimizer,
            step_size=config.training.lr_step_interval,
            gamma=config.training.lr_gamma,
        )

    def set_network_configs(self):
        self.config.network.body.indim = self.config.env.obs_dim
        self.config.network.heads["actor"].outdim = self.config.env.action_dim

    def discounted_advantages(self, rewards, masks, values):
        tau = self.config.algorithm.tau
        gamma = self.config.algorithm.gamma

        shape = values.shape
        returns = torch.zeros(shape).to(self.device)
        deltas = torch.zeros(shape).to(self.device)
        advantages = torch.zeros(shape).to(self.device)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0

        # Compute discounted returns and advantages
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i] + gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
            advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

            prev_return = returns[i]
            prev_value = values[i]
            prev_advantage = advantages[i]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)

        return advantages, returns

    def update(self):
        # Convert list to tensor
        states = torch.stack(self.memory.state).to(self.device)
        actions = torch.stack(self.memory.action).to(self.device)
        masks = torch.tensor(self.memory.mask).to(self.device)
        rewards = torch.tensor(self.memory.reward).to(self.device)
        old_logprobs = torch.stack(self.memory.logprob).to(self.device)
        with torch.no_grad():
            values = self.policy.critic_forward(states)
            advantages, returns = self.discounted_advantages(rewards, masks, values)

        # Optimize policy for K epochs:
        for _ in range(self.config.algorithm.optim_epochs):
            permutation = torch.randperm(states.shape[0]).to(self.device)

            for m in range(0, states.shape[0], self.config.training.minibatch_size):
                idxs = permutation[m : m + self.config.training.minibatch_size]

                logprobs, state_values, dist_entropy = self.policy.evaluate(
                    states[idxs], actions[idxs]
                )

                # Find the ratio (policy / old policy):
                ratios = torch.exp(logprobs - old_logprobs[idxs])

                # Compute surrogate loss
                surr1 = ratios * advantages[idxs]
                surr2 = (
                    torch.clamp(
                        ratios,
                        1 - self.config.algorithm.clip,
                        1 + self.config.algorithm.clip,
                    )
                    * advantages[idxs]
                )

                actor_loss = -torch.min(surr1, surr2)
                critic_loss = (state_values - returns[idxs]) ** 2
                #                 entropy_penalty = -0.01 * dist_entropy
                loss = actor_loss + critic_loss
                #                      + entropy_penalty
                # TODO: add back in the entropy penalty

                if self.config.costs.block_entropy:
                    blockH = block_entropy(
                        actions[idxs],
                        masks[idxs],
                        self.config.env.action_dim,
                        self.config.costs.max_block_length,
                        self.config.costs.sample_blocks,
                        self.config.costs.n_samples,
                    )
                    bH_loss = self.config.costs.block_ent_coeff * blockH
                    print("Block entropy: {}".format(blockH))
                    print("Block entropy loss: {}".format(bH_loss))
                    loss += blockH

                # Take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 40)
                self.optimizer.step()

        # Step learning rate
        self.lr_scheduler.step()

    def step(self, state):
        # Run old policy:
        ########################################
        # Hack for compatability with other mini-grid envs
        try:
            env_data = (
                self.env.get_data()
            )  # MC: possible off-by-one, but need to verify
        except AttributeError:
            env_data = None
        ########################################
        action, log_prob = self.policy.act(state)
        next_state, reward, done, _ = self.env.step(action.item())
        # self.env.render()
        if self.episode_steps == self.config.training.max_episode_length:
            done = True

        step_data = {
            "reward": reward,
            "mask": bool(not done),
            "state": state,
            "action": action,
            "logprob": log_prob,
            "env_data": env_data,
        }

        # Push to memory:
        self.memory.push(step_data)

        return step_data, next_state, done
