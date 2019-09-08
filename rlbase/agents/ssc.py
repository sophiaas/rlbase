import numpy as np
import torch
import torch.nn as nn
import os
import pickle

from agents import PPO
from core.analysis import load_episode_data
from core.replay_buffer import Memory
from utils.hierarchical_sparse_compressor import HierarchicalSparseCompressor


class SSC(PPO):
    
    def __init__(self, config):
        super(SSC, self).__init__(config)
        
        self.memory = Memory(features=['reward', 'mask', 'state', 'end_state', 
                                       'action', 'logprob', 'value', 'action_length',
                                       'env_data'])
        
        if os.path.exists(config.algorithm.load_dir+'action_dictionary.p'):
            with open(config.algorithm.load_dir+'action_dictionary.p', 'rb') as f:
                self.action_dictionary = pickle.load(f)
            self.config.network.heads['actor'].outdim = config.env.action_dim \
                                                        + config.algorithm.n_hl_actions  \
                                                        + len(self.action_dictionary)
        else:
            self.action_dictionary = {}
            self.config.network.heads['actor'].outdim = config.env.action_dim \
                                                        + config.algorithm.n_hl_actions 
        
        self.config.algorithm.n_actions = config.env.action_dim
        print('n actions: {}'.format(self.config.algorithm.n_actions))
        print('actor out dim: {}'.format(self.config.network.heads['actor'].outdim))
        
        if self.config.algorithm.load_action_dir is None:

            self.data = load_episode_data(config.algorithm.load_dir)
            print('action data: {}'.format(self.data.action))

            self.compressor = HierarchicalSparseCompressor(config.algorithm)
            self.compressor.compress(list(self.data.action))
            print('added motifs: {}'.format(self.compressor.added_motifs))

            self.action_dictionary = {**self.action_dictionary, **self.compressor.added_motifs}
            
        else:
            with open(self.config.algorithm.load_action_dir+'action_dictionary.p', 'rb') as f:
                learned_action_dic = pickle.load(f)
                self.action_dictionary = {**self.action_dictionary, **learned_action_dic}
        
        with open(self.logger.logdir+'action_dictionary.p', 'wb') as f:
            pickle.dump(self.action_dictionary, f)
            
        print(self.action_dictionary)
        
        
    def discounted_advantages(self, rewards, masks, values, action_lengths):
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
            
            if action_lengths[i] > 1:
                prev_return *= gamma ** (action_lengths[i] - 1)
                prev_value *= gamma ** (action_lengths[i] - 1)
                prev_advantage *= gamma ** (action_lengths[i] - 1)
                prev_advantage *= tau ** (action_lengths[i] - 1)
                
            returns[i] = rewards[i] + gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
            advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

            prev_return = returns[i]
            prev_value = values[i]
            prev_advantage = advantages[i]
                
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / advantages.std()
        
        return advantages, returns
        
        
    def step(self, state):
        # Run old policy:
        env_data = self.env.get_data()
        action, log_prob = self.policy.act(state)
        if action.item() < self.config.env.action_dim:
            next_state, reward, done, _ = self.env.step(action.item())

            if self.episode_steps == self.config.training.max_episode_length:
                done = True
            
            step_data = {
                'reward': reward, 
                'mask': bool(not done),
                'state': state,
                'action': action,
                'logprob': log_prob,
                'env_data': env_data,
                'action_length': 1
#                 'hl_mask': 1
            }
            
            self.memory.push(step_data)

        else:
            action_list = self.action_dictionary[action.item()]
#             first_step = True
            total_reward = 0
            for a in action_list:
                if done:
                    break
                next_state, reward, done, _ = self.env.step(a)
                if self.episode_steps == self.config.training.max_episode_length:
                    done = True
                total_reward += reward 
#                 if first_step:
#                     hl_mask = 1
#                 else: 
#                     hl_mask = 0
#                 step_data = {'reward': reward, 'hl_mask': hl_mask, 'mask': bool(not done)}
                
#                 self.memory.push(step_data)          
                
            step_data = {
                'reward': total_reward, 
                'mask': bool(not done),
                'state': state,
                'action': action,
                'logprob': log_prob,
                'env_data': env_data,
                'action_length': len(action_list)
            }
        
            self.memory.push(step_data) 
        
        return step_data, next_state, done
               
        
    def update(self):   
        # Convert list to tensor
        states = torch.stack(self.memory.state).to(self.device)
        actions = torch.stack(self.memory.action).to(self.device)
        masks = torch.tensor(self.memory.mask).to(self.device)
        rewards = torch.tensor(self.memory.reward).to(self.device)
        action_lengths = torch.tensor(self.memory.action_length).to(self.device)
        old_logprobs = torch.stack(self.memory.logprob).to(self.device)

        with torch.no_grad():
            values = self.policy.critic_forward(states)
            advantages, returns = self.discounted_advantages(rewards, masks, values, action_lengths)
                
        # Optimize policy for K epochs:
        for _ in range(self.config.algorithm.optim_epochs):
            permutation = torch.randperm(states.shape[0]).to(self.device)
            
            for m in range(0, states.shape[0], self.config.training.minibatch_size):
                idxs = permutation[m:m+self.config.training.minibatch_size]

                # Evaluate actions :
                logprobs, state_values, dist_entropy = self.policy.evaluate(
                                                                states[idxs], 
                                                                actions[idxs])

                # Find the ratio (policy / old policy):
                ratios = torch.exp(logprobs - old_logprobs[idxs])

                # Compute surrogate loss
                surr1 = ratios * advantages[idxs]
                surr2 = torch.clamp(ratios, 1-self.config.algorithm.clip, 
                                    1+self.config.algorithm.clip) * advantages[idxs]
                
                actor_loss = -torch.min(surr1, surr2)       
                critic_loss = (state_values - returns[idxs]) ** 2
#                 entropy_penalty = -0.01 * dist_entropy
                loss = actor_loss + critic_loss
#                      + entropy_penalty
                #TODO: add back in the entropy penalty

                # Take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 40)
                self.optimizer.step()
        
        # Step learning rate
        self.lr_scheduler.step()
