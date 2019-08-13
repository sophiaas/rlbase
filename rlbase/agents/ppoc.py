import numpy as np
import torch
import torch.nn as nn

from .base import BaseAgent
from networks.option_critic import OptionCritic
from core.replay_buffer import Memory
from envs import Lightbot


"""
Option-Critic trained with Proximal Policy Optimization
"""
torch.backends.cudnn.benchmark=True

class PPOC(BaseAgent):
    
    def __init__(self, config):
        super(PPOC, self).__init__(config)
        
        self.memory = Memory(features=['reward', 'mask', 'state', 'action',
                                       'action_logprob', 'option', 'option_logprob',
                                       'term_prob', 'terminate', 'env_data'])

        self.n_options = self.config.algorithm.n_options
        
        self.set_network_configs()
        
        self.policy = OptionCritic(config).to(self.device)
        self.policy_old = OptionCritic(config).to(self.device)

        self.optimizer = config.training.optim(self.policy.parameters(),
                                          lr=self.config.training.lr, 
                                          betas=self.config.training.betas)
        
        print('pp: {}'.format(next(self.policy.parameters()).is_cuda))
        print('po: {}'.format(next(self.policy_old.parameters()).is_cuda))
        
        self.terminated = True
        self.current_option = None

        self.MSELoss = nn.MSELoss()
        
    def set_network_configs(self):
        self.config.network.body.indim = self.config.env.obs_dim
        self.config.network.heads['actor'].outdim = self.config.env.action_dim
        self.config.network.heads['actor'].n_options = self.n_options
        self.config.network.heads['option_actor'].outdim = self.n_options
        self.config.network.heads['critic'].outdim = self.n_options
        self.config.network.heads['termination'].outdim = self.n_options
        
    def discount(self):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for i, reward in enumerate(reversed(self.memory.reward)):
            discounted_reward = reward + (self.config.algorithm.gamma * discounted_reward * self.memory.mask[i])
            rewards.insert(0, discounted_reward)
        return rewards
    
    def sample_blocks(self, sequence, max_length, n_samples):
        blocks = {x: [] for x in range(2, max_length+1)}
        for b in range(2, max_length+1):
            for i in range(n_samples):
#                 idx0 = np.random.randint(sequences.shape[0])
#                 idx1 = np.random.randint(sequences[idx0].shape[0]-max_length)
                idx = np.random.randint(sequence.shape[0]-max_length)
                random_block = tuple(sequence[idx:idx+b])
#                 random_block = tuple(sequences[idx0,idx1:idx1+b])
                blocks[b].append(random_block)            
        return blocks

    def block_entropy(self, sequence, possible_values):
        max_length = self.config.algorithm.max_block_length
        if self.config.algorithm.sample_blocks:
            blocks = self.sample_blocks(sequence, max_length, self.config.algorithm.n_block_samples)
        else:
            blocks = self.get_blocks(sequence, max_length)
        probs = {i: torch.zeros(size=[possible_values]*i) for i in range(2, max_split+1)}
        for d in range(2, max_split+1):
            for instance in blocks[d]:
                probs[d][instance] += 1
        distributions = [Categorical(x.view(-1)) for i,x in probs.items()]
        entropy = torch.tensor([x.entropy() for x in distributions])
        block_H = entropy.mean()
        return block_H
        
    def update(self):   
        # Normalizing the rewards:
        rewards = self.discount()
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)
        
        # convert list to tensor
        old_states = torch.stack(self.memory.state).to(self.device).detach()
        old_actions = torch.stack(self.memory.action).to(self.device).detach()
        old_action_logprobs = torch.stack(self.memory.action_logprob).to(self.device).detach()
        old_term_probs = torch.stack(self.memory.term_prob).to(self.device).detach()
        old_options = torch.stack(self.memory.option).to(self.device).detach()
        old_option_logprobs = torch.stack(self.memory.option_logprob).to(self.device).detach()
        
        
        # Optimize policy for K epochs:
        for _ in range(self.config.algorithm.optim_epochs):
            
            # Evaluating old actions and values :
            action_logprobs, option_values, option_values_full, option_logprobs, \
            option_probs, term_probs, action_dist_entropy,\
            option_dist_entropy \
            = self.policy.evaluate(old_states, old_actions, old_options)
            
#             print('OPTION VALS: {}'.format(option_values.shape))
#             print('OPTION LOG PROBS {}'.format(option_logprobs_full.shape))
#             print('times logprobs: {}'.format(torch.sum((option_values_full * option_logprobs_full), 1).shape))
#             print('dc: {}'.format(self.config.algorithm.dc))
        
            term_advantages = option_values.detach() \
                              - torch.sum((option_values_full * option_probs), 1) \
                              + self.config.algorithm.dc #TODO: change multiply and sum to mat mul? 
                            #should option_vals and option_probs be detached??
#                               - torch.sum((option_values_full.detach() * option_probs.detach()), 1) \

            
            # Finding Action Surrogate Loss:
            advantages = rewards - option_values.detach()
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(action_logprobs - old_action_logprobs.detach())
                               
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.config.algorithm.clip, 1+self.config.algorithm.clip) \
                                * advantages
                               
            # Finding Option Surrogate Loss:
            
            O_ratios = torch.exp(option_logprobs - old_option_logprobs.detach())
                               
            O_surr1 = O_ratios * advantages
            O_surr2 = torch.clamp(O_ratios, 1-self.config.algorithm.clip, 1+self.config.algorithm.clip) \
                                * advantages
                               
            actor_loss = -torch.min(surr1, surr2)
            option_actor_loss = -torch.min(O_surr1, O_surr2)
            critic_loss = (option_values - rewards) ** 2 
            term_loss = term_probs * term_advantages.unsqueeze(1)
            entropy_penalties = - (self.config.training.ent_coeff * action_dist_entropy 
                                + self.config.training.ent_coeff * option_dist_entropy)
            
#             print('actor_loss {}'.format(actor_loss.shape))
#             print('option actor loss {}'.format(option_actor_loss.shape))
#             print('critic_loss {}'.format(critic_loss.shape))
#             print('term loss {}'.format(term_loss.shape))
#             print('entropy_penalties {}'.format(entropy_penalties.shape))
            
            loss = actor_loss + option_actor_loss + critic_loss + term_loss + entropy_penalties
        
            if self.config.algorithm.block_ent_penalty:
                block_entropy = self.block_ent_coeff * self.block_entropy(old_actions, self.config.env.action_dim)
                loss += block_entropy

            #TODO: separate term_loss and others (does this really matter tho?)
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
    def step(self, state):
        # Running policy_old:
        start_state, action, action_logprob, option, option_logprob, \
                    term_prob, terminate = self.policy_old.act(state, self.current_option)
        state, reward, done, env_data = self.env.step(action.item())
        self.current_option = option.data
        
        step_data = {
            'reward': reward, 
            'mask': bool(not done),
            'state': start_state,
            'action': action,
            'action_logprob': action_logprob,
            'option': option,
            'option_logprob': option_logprob,
            'term_prob': term_prob,
            'terminate': terminate,
            'env_data': env_data
        }
        
        # Push to memory:
        self.memory.push(step_data)
        
        return step_data, state, done
