import numpy as np
import torch
import torch.nn as nn
import os
import pickle
import copy

from agents import PPO
from core.analysis import load_episode_data
from core.replay_buffer import Memory
from utils.hierarchical_sparse_compressor import HierarchicalSparseCompressor
from networks.actor_critic import ActorCritic

from collections import defaultdict



class SSC(PPO):
    
    def __init__(self, config):
        super(SSC, self).__init__(config)
        
        self.memory = Memory(features=['reward', 'mask', 'state', 'end_state', 
                                       'action', 'logprob', 'value', 'action_length',
                                       'env_data'])
        
        if os.path.exists(config.algorithm.load_dir+'action_dictionary.p'):
            with open(config.algorithm.load_dir+'action_dictionary.p', 'rb') as f:
                self.action_dictionary = pickle.load(f)
            full_action_dim = config.env.action_dim + config.algorithm.n_hl_actions + len(self.action_dictionary)
            self.config.network.heads['actor'].outdim = full_action_dim
            self.config.algorithm.n_actions = full_action_dim                                                                       
        else:
            self.action_dictionary = {}
            self.config.network.heads['actor'].outdim = config.env.action_dim \
                                                        + config.algorithm.n_hl_actions
            self.config.algorithm.n_actions = self.config.env.action_dim  
            
        self.policy = ActorCritic(config).to(self.device)

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
        
        for i in reversed(range(len(rewards))):
            if action_lengths[i] > 1:
                if len(rewards[i]) < action_lengths[i]:
                    diff = action_lengths[i].cpu().data.numpy() - len(rewards[i])
                    rewards[i] += [0] * diff
                discounted_rewards = [rewards[i][x] * gamma ** x for x in range(action_lengths[i])]
                r = torch.tensor(np.sum(discounted_rewards))
            else:
                r = torch.tensor(rewards[i])
                
            returns[i] = r + gamma * prev_return * masks[i]
            deltas[i] = r + gamma * prev_value * masks[i] - values[i]
            advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

            prev_return = returns[i]
            prev_value = values[i]
            prev_advantage = advantages[i]
                
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / advantages.std()
        
        return advantages, returns
    
    def test_hl_action(self, state, hl_action):
        valid = True
        state = copy.deepcopy(self.env.raw_state)
        for i, a in enumerate(hl_action):
            original_state = copy.deepcopy(state)
            state, test_reward, test_done = self.env.make_move(original_state, a, test=True)
            if state == original_state:
                valid = False
                break  
            if test_done and i < len(hl_action) - 1:
                valid = False
                break
        return valid
    
    def uncompress_hl_action(self, hl_action):
        n_primitives = self.config.env.action_dim
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
            }
            
            self.memory.push(step_data)

        else:
            action_list = self.action_dictionary[action.item()]
            hl_action = self.uncompress_hl_action(action_list)
            valid = self.test_hl_action(state, hl_action)
            total_reward = []
            done = False
            if valid:
                for a in hl_action:
                    if done:
                        break
                    next_state, reward, done, _ = self.env.step(a)
                    total_reward.append(reward)

                    if self.episode_steps == self.config.training.max_episode_length:
                        done = True
            else:
                #NB: Hard coded!! This is not good, but will be changed later
                total_reward = [-1] * len(action_list) 
                next_state = state.cpu().data.numpy()
                
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
        
        
    def train(self):
        # TODO: Add handle resumes
        running_reward = 0
        avg_length = 0
        timestep = 0
        self.episode_steps = 0

        # Iterate through episodes
        for i_episode in range(1, self.config.training.max_episodes+1):
            episode_reward = 0
            episode_data = defaultdict(list, {'episode': int(self.episode)})
            state = self.env.reset()
            
            # Iterate through steps
            action_tracker = 1
            for t in range(1, self.config.training.max_episode_length+1):
                if action_tracker == self.config.training.max_episode_length:
                    break
                state = torch.from_numpy(state).float().to(self.device)
                
                with torch.no_grad():
                    transition, state, done = self.step(state)
                    
                timestep += transition['action_length']
                action_tracker += transition['action_length']
                self.episode_steps += transition['action_length']

                for key, val in transition.items():
                    episode_data[key].append(self.convert_data(val))
                
                if type(transition['reward']) == list:
                    rew = np.sum(transition['reward'])
                else:
                    rew = transition['reward']
                               
                episode_reward += rew
                running_reward += rew
                
                if self.config.experiment.render:
                    self.env.render()
                    
                if done:
                    break

                if timestep % self.config.training.update_every == 0:
                    self.update()
                    self.memory.clear()
            
            if self.config.experiment.save_episode_data and self.episode % self.config.experiment.every_n_episodes == 0:
                self.logger.push_episode_data(episode_data)
                
                
            # Logging
            if i_episode % self.config.experiment.log_interval == 0:
                
                print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, 
                                                                    round(self.running_moves, 2), 
                                                                    round(self.running_rewards, 2)))
                self.logger.save()
                self.logger.save_checkpoint(self)
                
                if self.config.experiment.save_episode_data:
                    self.logger.save_episode_data()
                
                self.logger.plot('running_rewards')
                self.logger.plot('running_moves')
                
            avg_length += self.episode_steps
            
            if i_episode % 10 == 0:
                self.update_running(running_reward/10, avg_length/10)
                running_reward = 0
                avg_length = 0
                self.logger.push(self.get_summary())
                
            self.episode_steps = 0
            self.episode += 1
            
        print('Training complete')
