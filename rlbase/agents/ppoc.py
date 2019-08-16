import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

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
        
        self.memory = Memory(features=['reward', 'mask', 'state', 'end_state', 
                                       'action', 'action_logprob', 'option', 
                                       'option_logprob', 'term_prob', 'terminate', 
                                       'env_data'])

        self.n_options = self.config.algorithm.n_options
        
        self.set_network_configs()
        
        self.policy = OptionCritic(config).to(self.device)
        self.policy_old = OptionCritic(config).to(self.device)

        self.optimizer = config.training.optim(self.policy.parameters(),
                                          lr=self.config.training.lr, 
                                          betas=self.config.training.betas,
                                          weight_decay=config.training.weight_decay)
        self.lr_scheduler = self.config.training.lr_scheduler(self.optimizer, 
                                                              step_size=1, 
                                                              gamma=config.training.lr_gamma)
        
        self.terminated = True
        self.current_option = None
        
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
    
    def get_blocks(sequence, masks, max_length):
        #TODO: make sure masking is working
        blocks = {x: [] for x in range(2, max_length+1)}
        for i in range(2, max_length+1):
            m = [tuple(masks[a:a+i]) for a in range(len(masks)-i)]
            exclude = [x for x in m if (x==0).nonzero().shape[0] > 0]
            blocks[i] += [tuple(sequence[a:a+i]) for a in range(len(sequence)-i) if a not in exclude]
        return blocks
    
    def sample_blocks(self, sequence, masks, max_length, n_samples):
        #TODO: make sure masking is working
        blocks = {x: [] for x in range(2, max_length+1)}
        episode_ends = (masks==0).nonzero()
        for b in range(2, max_length+1):
            for i in range(n_samples):
                nonvalid = []
                for end in episode_ends:
                    nonvalid += [end-x for x in range(b+1)]
                idx_set = [x for x in range(sequence.shape[0]-max_length) if x not in nonvalid]
                idx = np.random.choice(idx_set)
                random_block = tuple(sequence[idx:idx+b])
                blocks[b].append(random_block)            
        return blocks

    def block_entropy(self, sequence, masks, possible_values):
        max_length = self.config.algorithm.max_block_length
        if self.config.algorithm.sample_blocks:
            blocks = self.sample_blocks(sequence, masks, max_length, self.config.algorithm.n_block_samples)
        else:
            blocks = self.get_blocks(sequence, max_length)
        probs = {i: torch.zeros(size=[possible_values]*i) for i in range(2, max_length+1)}
        for d in range(2, max_length+1):
            for instance in blocks[d]:
                probs[d][instance] += 1
        distributions = [Categorical(x.view(-1)) for i,x in probs.items()]
        entropy = torch.tensor([x.entropy() for x in distributions])
        block_H = entropy.mean()
        return block_H, entropy
        
    def update(self):   
        # Normalizing the rewards:
        rewards = self.discount()
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)
        
        # convert list to tensor
        old_states = torch.stack(self.memory.state).to(self.device).detach()
        old_masks = torch.tensor(self.memory.mask).to(self.device).detach()
#         old_masks = torch.stack(self.memory.mask).to(self.device).detach()
        old_actions = torch.stack(self.memory.action).to(self.device).detach()
        old_action_logprobs = torch.stack(self.memory.action_logprob).to(self.device).detach()
        old_term_probs = torch.stack(self.memory.term_prob).to(self.device).detach()
        old_options = torch.stack(self.memory.option).to(self.device).detach()
        old_option_logprobs = torch.stack(self.memory.option_logprob).to(self.device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.config.algorithm.optim_epochs):
            permutation = torch.randperm(old_states.shape[0]).to(self.device)
            
            for m in range(0, old_states.shape[0], self.config.training.minibatch_size):
                idxs = permutation[m:m+self.config.training.minibatch_size]
            
                # Evaluating old actions and values :
                action_logprobs, option_values, option_values_full, option_logprobs, \
                option_probs, term_probs, action_dist_entropy,\
                option_dist_entropy \
                = self.policy.evaluate(old_states[idxs], old_actions[idxs], old_options[idxs])

                term_advantages = option_values.detach() \
                                  - torch.sum((option_values_full.detach() * option_probs.detach()), 1) \
                                  + self.config.algorithm.dc #TODO: change multiply and sum to mat mul? 
                                # TODO: should option_vals and option_probs be detached??
    #                               - torch.sum((option_values_full.detach() * option_probs.detach()), 1) \


                # Finding Action Surrogate Loss:
                advantages = rewards[idxs] - option_values.detach()

                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(action_logprobs - old_action_logprobs.detach()[idxs])

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.config.algorithm.clip, 1+self.config.algorithm.clip) \
                                    * advantages

                # Finding Option Surrogate Loss:

                O_ratios = torch.exp(option_logprobs - old_option_logprobs.detach()[idxs])

                O_surr1 = O_ratios * advantages
                O_surr2 = torch.clamp(O_ratios, 
                                      1-self.config.algorithm.clip, 
                                      1+self.config.algorithm.clip) * advantages

                actor_loss = -torch.min(surr1, surr2)
                option_actor_loss = -torch.min(O_surr1, O_surr2)
                critic_loss = (option_values - rewards[idxs]) ** 2 
                term_loss = term_probs * term_advantages.unsqueeze(1)
                entropy_penalties = -(self.config.training.ent_coeff * action_dist_entropy 
                                    + self.config.training.ent_coeff * option_dist_entropy)

                loss = actor_loss + option_actor_loss + critic_loss + term_loss + entropy_penalties

                if self.config.algorithm.block_ent_penalty:
                    block_entropy, e = self.block_entropy(old_actions[idxs], 
                                                          old_masks[idxs], 
                                                          self.config.env.action_dim[idxs])
                    be_loss = self.config.algorithm.block_ent_coeff * block_entropy
                    print('BLOCK ENTROPY: {}'.format(block_entropy))
                    print('E: {}'.format(e))
                    print('be loss: {}'.format(be_loss))
                    loss += block_entropy

                #TODO: separate term_loss and others (does this really matter tho?)

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 40)
                self.optimizer.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Step learning rate
        self.lr_scheduler.step()
        
    def step(self, state):
        # Running policy_old:
        start_state, action, action_logprob, start_option, next_option, \
        option_logprob, term_prob, terminate = self.policy_old.act(state, self.current_option)
        state, reward, done, env_data = self.env.step(action.item())
        
        
        step_data = {
            'reward': reward, 
            'mask': bool(not done),
            'state': start_state,
            'end_state': state,
            'action': action,
            'action_logprob': action_logprob,
            'option': start_option,
            'option_logprob': option_logprob,
            'term_prob': term_prob,
            'terminate': terminate,
            'env_data': env_data
        }
        #TODO: This should fix the option-state coordination problem but verify
        
        self.current_option = next_option.data
        
        # Push to memory:
        self.memory.push(step_data)
        
        return step_data, state, done
