import numpy as np
import torch
import torch.nn as nn
import os
import pickle
import copy

# from agents import PPO, BaseAgent
from agents.base import BaseAgent
from core.analysis import load_episode_data
from core.replay_buffer import Memory
from utils.hierarchical_sparse_compressor import HierarchicalSparseCompressor
from networks.actor_critic import ActorCritic

from collections import defaultdict



class SSC(BaseAgent):
    
    def __init__(self, config):
        super(SSC, self).__init__(config)
        
        self.memory = Memory(features=['reward', 'mask', 'state', 'end_state', 
                                       'action', 'logprob', 'value', 'action_length',
                                       'env_data'])
        
        self.config.network.body.indim = self.config.env.obs_dim
        
        if config.algorithm.load_dir and os.path.exists(config.algorithm.load_dir+'action_dictionary.p'):
            with open(config.algorithm.load_dir+'action_dictionary.p', 'rb') as f:
                self.action_dictionary = pickle.load(f)
                self.config.algorithm.n_actions = config.env.action_dim + len(self.action_dictionary)
#             else:
#                 full_action_dim = config.env.action_dim + config.algorithm.n_hl_actions
#                 self.config.algorithm.n_actions = config.env.action_dim

#             self.config.network.heads['actor'].outdim = full_action_dim
                                                                               
        else:
            self.action_dictionary = {}
#             self.config.network.heads['actor'].outdim = config.env.action_dim \
#                                                         + config.algorithm.n_hl_actions
            self.config.algorithm.n_actions = self.config.env.action_dim  
    
        self.config.network.heads['actor'].outdim = self.config.algorithm.max_actions
        self.available_actions = self.config.algorithm.n_actions + self.config.algorithm.n_hl_actions
            
        self.policy = ActorCritic(config).to(self.device)

        self.optimizer = config.training.optim(self.policy.parameters(),
                                          lr=self.config.training.lr) 

        self.lr_scheduler = self.config.training.lr_scheduler(
                                        self.optimizer, 
                                        step_size=config.training.lr_step_interval, 
                                        gamma=config.training.lr_gamma)

        print('n actions: {}'.format(self.config.algorithm.n_actions))
        print('actor out dim: {}'.format(self.config.network.heads['actor'].outdim))
        
        if self.config.algorithm.load_action_dir is None and self.config.algorithm.load_dir is not None:

            self.data = load_episode_data(config.algorithm.load_dir)
            print('lengt of aciton data: {}'.format(len(self.data)))
            print('action data: {}'.format(list(self.data.action)))
            if self.env.name == 'hanoi':
                action_data = []
                for x in list(self.data.action):
                    action_data += x
                action_data = [action_data]
            else:
                action_data = list(self.data.action)

            self.compressor = HierarchicalSparseCompressor(config.algorithm)
            self.compressor.compress(action_data)
            print('added motifs: {}'.format(self.compressor.added_motifs))

            self.action_dictionary = {**self.action_dictionary, **self.compressor.added_motifs}
            
        elif self.config.algorithm.load_action_dir:
            with open(self.config.algorithm.load_action_dir+'action_dictionary.p', 'rb') as f:
                learned_action_dic = pickle.load(f)
                self.action_dictionary = {**self.action_dictionary, **learned_action_dic}
        
        with open(self.logger.logdir+'action_dictionary.p', 'wb') as f:
            pickle.dump(self.action_dictionary, f)
            
        print(self.action_dictionary)
        print(self.policy)
        
        
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
        
        for i in reversed(range(len(rewards))):
            
            if action_lengths[i] > 1:
                
                if len(rewards[i]) < action_lengths[i]:
                    diff = action_lengths[i].cpu().data.numpy() - len(rewards[i])
                    rewards[i] += [0] * diff
                    
                discounted_rewards = [rewards[i][x] * gamma ** x for x in range(action_lengths[i])]
                r = torch.tensor(np.sum(discounted_rewards))
                prev_return *= gamma ** (action_lengths[i] - 1)
        
            else:
                r = rewards[i]
                
            returns[i] = r + gamma * prev_return * masks[i]
            deltas[i] = r + gamma * prev_value * masks[i] - values[i]
            advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

            prev_return = returns[i]
            prev_value = values[i]
            prev_advantage = advantages[i]
                
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / advantages.std()
        
        return advantages, returns

    
    def uncompress_hl_action(self, hl_action):
        n_primitives = self.config.algorithm.n_actions
        primitive = False if np.any([x >= n_primitives for x in hl_action]) else True
        while not primitive:
            primitive_sequence = []
            for x in hl_action:
                if x < n_primitives:
                    primitive_sequence.append(x)
                else: 
                    primitive_sequence += self.action_dictionary[x]
            hl_action = primitive_sequence
            primitive = False if np.any([x >= n_primitives for x in hl_action]) else True
        return hl_action

        
    def step(self, state):
        # Run old policy:
        env_data = self.env.get_data()
#         action = self.available_actions
#         while action >= self.available_actions:
            
        action, log_prob = self.policy.act(state, cutoff=self.available_actions)
#             if action.item() == 6:
#                 print(action, log_prob)
        if action.item() < self.config.env.action_dim:
            next_state, reward, done, _ = self.env.step(action.item())
            self.episode_steps += 1
#             if self.episode_steps == self.config.training.max_episode_length:
#                 done = True

            step_data = {
                'reward': reward, 
                'mask': 0 if done else 1,
                'state': state,
                'action': action,
                'logprob': log_prob,
                'env_data': env_data,
                'action_length': 1
            }

            self.memory.push(step_data)
            
        else:
            action_list = self.action_dictionary[action.item()]
            hl_action = self.uncompress_hl_action(action_list)
                
            total_reward = []
            done = False
            for a in hl_action:
                if done:
                    break
                                
                next_state, reward, done, _ = self.env.step(a)
                
                total_reward.append(reward)
                self.episode_steps += 1
                
                if self.episode_steps == self.config.training.max_episode_length:
                    done = True
                
            step_data = {
                'reward': total_reward, 
                'mask': 0 if done else 1,
                'state': state,
                'action': action,
                'logprob': log_prob,
                'env_data': env_data,
                'action_length': len(hl_action)
            }
        
            self.memory.push(step_data) 
        
        return step_data, next_state, done
 

    def update(self):   
        # Convert list to tensor
        states = torch.stack(self.memory.state).to(self.device)
        actions = torch.stack(self.memory.action).to(self.device)
        masks = torch.tensor(self.memory.mask).to(self.device)
        rewards = self.memory.reward
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
        
    def sample_episode(self, episode, step, run_avg):
        episode_return = 0
        self.episode_steps = 0
        episode_data = defaultdict(list, {'episode': int(episode)})
        state = self.env.reset()
        t = 0
        while t < self.config.training.update_every:
#         for t in range(self.config.training.update_every):
            state = torch.from_numpy(state).float().to(self.device)
            with torch.no_grad():
                transition, state, done = self.step(state)
            for key, val in transition.items():
                episode_data[key].append(self.convert_data(val))
            if type(transition['reward']) == list:
                rew = np.sum(transition['reward'])
            else:
                rew = transition['reward']
            episode_return += rew
            t += transition['action_length']
            if self.config.experiment.render:
                self.env.render()
            if (step+t+1) % self.config.experiment.num_steps_between_plot == 0:
                summary = {
                    'steps': step+t+1,
                    'return': run_avg.get_value('return'),
                    'moves': run_avg.get_value('moves')
                }
                self.logger.push(summary)
                # print('Pushed summary at step: {}'.format(step+t+1))
            if done:
                break
            self.episode_steps += transition['action_length']
        episode_length = t+1
        return episode_data, episode_return, episode_length
        
        
#     def train(self):
#         # TODO: Add handle resumes
#         timestep = 0
#         lower_timestep = 0
#         total_lower_timestep = 0
#         self.episode_steps = 0
        
#         episode_reward = 0
#         episode_data = defaultdict(list, {'episode': int(self.episode)})
#         state = self.env.reset()

#         # Iterate through timesteps
#         print('Max Timesteps: {}'.format(self.config.training.max_timesteps))
#         for timestep in range(1, self.config.training.max_timesteps+1):

#             self.episode_steps += 1
#             state = torch.from_numpy(state).float().to(self.device)

#             with torch.no_grad():
#                 transition, state, done = self.step(state)
                
#             for key, val in transition.items():
#                 episode_data[key].append(self.convert_data(val))

#             if type(transition['reward']) == list:
#                 rew = np.sum(transition['reward'])
#             else:
#                 rew = transition['reward']
                
#             episode_reward += rew
#             lower_timestep += transition['action_length']
#             total_lower_timestep += transition['action_length']


#             if self.config.experiment.render:
#                 self.env.render()
                    
#             if self.config.experiment.save_episode_data and self.episode % self.config.experiment.every_n_episodes == 0:
#                 self.logger.push_episode_data(episode_data)
                
# #             if self.episode_steps == self.config.experiment.max_episode_length:
# #                     done = True
                
#             if done:
#                 episode_data = defaultdict(list, {'episode': int(self.episode)})
#                 summary = {
#                     'steps': total_lower_timestep,
#                     'return': episode_reward,
#                     'moves': self.episode_steps
#                 }
#                 self.logger.push(summary)
#                 state = self.env.reset()
                
#                 # Logging
#                 if self.episode % self.config.experiment.log_interval == 0:
                    
#                     print('Episode {} \t Length: {} \t Reward: {}'.format(self.episode, 
#                                                                         round(self.episode_steps, 2), 
#                                                                         round(episode_reward, 2)))

#                     self.logger.save()
#                     self.logger.save_checkpoint(self)

#                     if self.config.experiment.save_episode_data:
#                         self.logger.save_episode_data(self.episode)

#                     self.logger.plot('return')
#                     self.logger.plot('moves')
#                 self.episode += 1
#                 self.episode_steps = 0
#                 episode_reward = 0

#             if lower_timestep >= self.config.training.update_every:
#                 self.update()
#                 self.memory.clear()
#                 lower_timestep = 0
                
# #             if timestep % self.config.training.update_every == 0:
# #                 self.update()
# #                 self.memory.clear()
                
#             if total_lower_timestep >= self.config.training.max_timesteps:
#                 break
            
#         print('Training complete')
        
    def evaluate(self):
        # TODO: Add handle resumes
        timestep = 0
        lower_timestep = 0
        total_lower_timestep = 0
        self.episode_steps = 0
        
        episode_reward = 0
        episode_data = defaultdict(list, {'episode': int(self.episode)})
        state = self.env.reset()

        # Iterate through timesteps
        print('Max Timesteps: {}'.format(self.config.training.max_timesteps))
        for timestep in range(1, self.config.training.max_timesteps+1):

            self.episode_steps += 1
            state = torch.from_numpy(state).float().to(self.device)

            with torch.no_grad():
                transition, state, done = self.step(state)
                
            for key, val in transition.items():
                episode_data[key].append(self.convert_data(val))

            if type(transition['reward']) == list:
                rew = np.sum(transition['reward'])
            else:
                rew = transition['reward']
                
            episode_reward += rew
            lower_timestep += transition['action_length']
            total_lower_timestep += transition['action_length']


            if self.config.experiment.render:
                self.env.render()
                    
            if self.config.experiment.save_episode_data and self.episode % self.config.experiment.every_n_episodes == 0:
                self.logger.push_episode_data(episode_data)
                
            if done:
                episode_data = defaultdict(list, {'episode': int(self.episode)})
                summary = {
                    'steps': total_lower_timestep,
                    'return': episode_reward,
                    'moves': self.episode_steps
                }
                self.logger.push(summary)
                state = self.env.reset()
                
                # Logging
                if self.episode % self.config.experiment.log_interval == 0:
                    
                    print('Episode {} \t Length: {} \t Reward: {}'.format(self.episode, 
                                                                        round(self.episode_steps, 2), 
                                                                        round(episode_reward, 2)))

                    self.logger.save()
                    self.logger.save_checkpoint(self)

                if self.config.experiment.save_episode_data:
                    self.logger.save_episode_data(self.episode)

                    self.logger.plot('return')
                    self.logger.plot('moves')
                self.episode += 1
                self.episode_steps = 0
                episode_reward = 0

            if timestep == self.config.eval.n_eval_steps:
                break
                
#             if timestep % self.config.training.update_every == 0:
#                 self.update()
#                 self.memory.clear()
                
            
        print('Training complete')
        

