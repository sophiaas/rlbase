import numpy as np
import torch
import torch.nn as nn

from .base import BaseAgent
from networks.actor_critic import ActorCritic
from core.replay_buffer import Memory
from envs import Lightbot

from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

torch.autograd.set_detect_anomaly(True)


"""
Advantage Actor-Critic Proximal Policy Optimization
"""

class PPO(BaseAgent):
    
    def __init__(self, config):
        super(PPO, self).__init__(config)
        
        self.config.network.body.indim = self.config.env.obs_dim
        self.config.network.heads['actor'].outdim = self.config.env.action_dim
        
        self.policy = ActorCritic(config).to(self.device)
        self.policy_old = ActorCritic(config).to(self.device)

        self.actor_optimizer = config.training.optim(self.policy.actor.parameters(),
                                  lr=self.config.training.lr, 
                                  betas=self.config.training.betas,
                                  weight_decay=self.config.training.weight_decay)
        self.critic_optimizer = config.training.optim(self.policy.critic.parameters(),
                                  lr=self.config.training.lr, 
                                  betas=self.config.training.betas,
                                  weight_decay=self.config.training.weight_decay)
        
        self.optimizer = config.training.optim(self.policy.parameters(),
                                          lr=self.config.training.lr, 
                                          betas=self.config.training.betas,
                                          weight_decay=self.config.training.weight_decay)
        
        self.lr_scheduler = self.config.training.lr_scheduler(self.actor_optimizer, 
                                                              step_size=1, 
                                                              gamma=config.training.lr_gamma)

#         self.actor_lr_scheduler = self.config.training.lr_scheduler(self.actor_optimizer, 
#                                                               step_size=1, 
#                                                               gamma=config.training.lr_gamma)
#         self.critic_lr_scheduler = self.config.training.lr_scheduler(self.critic_optimizer, 
#                                                               step_size=1, 
#                                                               gamma=config.training.lr_gamma)
        
    def discount(self):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for i, reward in enumerate(reversed(self.memory.reward)):
            discounted_reward = reward \
                                + (self.config.algorithm.gamma * discounted_reward * self.memory.mask[i])
            rewards.insert(0, discounted_reward)
        return rewards
        
    def update(self):   
        # Discount and normalize the rewards:
        rewards = self.discount()
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)
        
        # Convert list to tensor
#         old_states = torch.stack(self.memory.state).to(self.device).detach()
#         old_actions = torch.stack(self.memory.action).to(self.device).detach()
#         old_logprobs = torch.stack(self.memory.logprob).to(self.device).detach()  
        
        old_states = torch.stack(self.memory.state).to(self.device).detach().clone()
        old_actions = torch.stack(self.memory.action).to(self.device).detach().clone()
        old_logprobs = torch.stack(self.memory.logprob).to(self.device).detach().clone()

        
        with torch.no_grad():
            values = self.policy.critic_forward(old_states)[0]
            fixed_logprobs = self.policy.log_prob(old_states, torch.tensor(old_actions, 
                                                                          requires_grad=True, 
                                                                    dtype=torch.float).to(self.device))
#             fixed_logprobs, values, entropy = self.policy.evaluate(old_states, torch.tensor(old_actions, requires_grad=True, dtype=torch.float).to(self.device))
#         print('rewards shape {}'.format(rewards.shape))
#         print('values shape {}'.format(values.shape))
        advantages = rewards - values
                    
        # Optimize policy for K epochs:
        for _ in range(self.config.algorithm.optim_epochs):
            permutation = torch.randperm(old_states.shape[0]).to(self.device)
            for m in range(0, old_states.shape[0], self.config.training.minibatch_size):
                idxs = permutation[m:m+self.config.training.minibatch_size]

#                 # Evaluate old actions and values :
#                 logprobs, state_values, dist_entropy = self.policy.evaluate(old_states[idxs], old_actions[idxs])
#                 logprobs, values_pred, dist_entropy = self.policy.evaluate(torch.tensor(old_states[idxs], requires_grad=True).to(self.device), torch.tensor(old_actions[idxs], dtype=torch.float, requires_grad=True).to(self.device))
                values_pred = self.policy.critic_forward(torch.tensor(old_states[idxs], 
                                                                     requires_grad=True).to(self.device))
                critic_loss = (values_pred - torch.tensor(rewards[idxs], 
                                                          requires_grad=True).to(self.device)) ** 2


#                 # Find the ratio (policy / old policy):
#                 ratios = torch.exp(logprobs - old_logprobs[idxs])


#                 # Find surrogate loss:
#                 #TODO: this is different than the advantage calculation in old code. Find out if it matters
#                 advantages = rewards[idxs] - state_values.detach()
#                 advantages = rewards[idxs] - state_values
#                 advantages = torch.tensor(rewards[idxs] - values[idxs], requires_grad=True).to(self.device)
                               
                log_probs = self.policy.log_prob(torch.tensor(old_states[idxs], 
                                                              requires_grad=True).to(self.device), 
                                                torch.tensor(old_actions[idxs], dtype=torch.float, 
                                                             requires_grad=True).to(self.device))

                ratios = torch.exp(log_probs - torch.tensor(fixed_logprobs[idxs], 
                                                            requires_grad=True).to(self.device))
                advantages_var = torch.tensor(advantages[idxs], requires_grad=True).to(self.device)

                surr1 = ratios * advantages_var
                surr2 = torch.clamp(ratios, 1-self.config.algorithm.clip, 1+self.config.algorithm.clip) \
                                    * advantages_var
                actor_loss = -torch.min(surr1, surr2) 
#                 critic_loss = (values_pred - rewards[idxs]) ** 2

#                 critic_loss = (state_values - rewards[idxs]) ** 2
#                 entropy_penalty = -0.01 * dist_entropy
#                 loss = actor_loss + critic_loss + entropy_penalty
                loss = actor_loss + critic_loss

#                 # Take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 40)
                self.optimizer.step()



        self.lr_scheduler.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
    def step(self, state):
        # Run old policy:
        action, log_prob, value, start_state = self.policy_old.act(state)
#         action, start_state, log_prob, value = 
        state, reward, done, env_data = self.env.step(action.item())
        
        step_data = {
            'reward': reward, 
            'mask': bool(not done),
            'state': start_state,
            'action': action,
            'value': value,
            'logprob': log_prob,
            'env_data': env_data
        }
        
        # Push to memory:
        self.memory.push(step_data)
        
        return step_data, state, done
