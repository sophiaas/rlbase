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

        self.optimizer = config.training.optim(self.policy.parameters(),
                                          lr=self.config.training.lr)
        self.lr_scheduler = self.config.training.lr_scheduler(
                                            self.optimizer, 
                                            step_size=config.training.lr_step_interval, 
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
        
        #IMPORTANT TODO: If return = 0, gamma is not exponentiated--eh maybe not? just think about it
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
            blocks = self.sample_blocks(sequence, masks, max_length, 
                                        self.config.algorithm.n_block_samples)
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
    
    def term_advantages(self, option_values, option_values_full, option_log_probs):
        #TODO: should this be option logprobs or logprobs?        
        advantages = option_values - torch.sum(torch.mul(option_values_full, option_log_probs), 1) \
                        + torch.tensor(self.config.algorithm.dc).to(self.device)
        return advantages
        
    def update(self):   
        
        # Convert list to tensor
        states = torch.stack(self.memory.state).to(self.device)
        actions = torch.stack(self.memory.action).to(self.device)
        options = torch.stack(self.memory.option).to(self.device)
        masks = torch.tensor(self.memory.mask).to(self.device)
        rewards = torch.tensor(self.memory.reward).to(self.device)
        
        old_action_log_probs = torch.stack(self.memory.action_logprob).to(self.device)
        old_option_log_probs = torch.stack(self.memory.option_logprob).to(self.device)
        old_term_probs = torch.stack(self.memory.term_prob).to(self.device)
        
        with torch.no_grad():
            old_option_values_full = self.policy.critic_forward(states)
            old_option_values = torch.cat([torch.index_select(a, 0, i) for a, i in zip(old_option_values_full, options)])
            advantages, returns = self.discounted_advantages(rewards, masks, old_option_values)
            old_option_log_probs_full = self.policy.option_logprobs_full(states)
            term_advantages = self.term_advantages(old_option_values, old_option_values_full, old_option_log_probs_full)
        
        # Optimize policy for K epochs:
        for _ in range(self.config.algorithm.optim_epochs):
            permutation = torch.randperm(states.shape[0]).to(self.device)
            
            for m in range(0, states.shape[0], self.config.training.minibatch_size):
                idxs = permutation[m:m+self.config.training.minibatch_size]
                #TODO: make this less arbitrary
                if idxs.shape[0] > self.config.training.minibatch_size / 5:
                            
                    # Evaluating old actions and values :
                    action_logprobs = self.policy.evaluate_action(states[idxs], actions[idxs], options[idxs])
                    option_logprobs = self.policy.evaluate_option(states[idxs], options[idxs])
                    option_values = self.policy.critic_forward(states[idxs], options[idxs])
                    term_probs = self.policy.term_forward(states[idxs], options[idxs])

                    # TODO: Should term advantages be computed inside or outside loop?

                    # Finding Action Surrogate Loss:

                    ratios = torch.exp(action_logprobs - old_action_log_probs[idxs])
                    surr1 = ratios * advantages[idxs]
                    surr2 = torch.clamp(ratios, 1-self.config.algorithm.clip, 
                                        1+self.config.algorithm.clip) * advantages[idxs]

                    # Finding Option Surrogate Loss:
                    O_ratios = torch.exp(option_logprobs - old_option_log_probs[idxs])
                    O_surr1 = O_ratios * advantages[idxs]
                    O_surr2 = torch.clamp(O_ratios, 
                                          1-self.config.algorithm.clip, 
                                          1+self.config.algorithm.clip) * advantages[idxs]

                    actor_loss = -torch.min(surr1, surr2)
                    option_actor_loss = -torch.min(O_surr1, O_surr2)
                    critic_loss = (option_values - returns[idxs]) ** 2 
                    term_loss = term_probs * term_advantages.unsqueeze(1)[idxs]

    #                 entropy_penalties = -(self.config.training.ent_coeff * action_dist_entropy 
    #                                     + self.config.training.ent_coeff * option_dist_entropy)

                    loss = actor_loss + option_actor_loss + critic_loss + term_loss

                    if self.config.algorithm.block_ent_penalty:
                        block_entropy, e = self.block_entropy(actions[idxs], 
                                                              masks[idxs], 
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
        
        # Step learning rate
        self.lr_scheduler.step()
        
    def step(self, state):
        # Running policy_old:
        env_data = self.env.get_data()
        # TODO: unpack
        action, option, next_option, term_prob, terminate, action_log_prob, option_log_prob = self.policy.act(state, self.current_option)
        next_state, reward, done, _ = self.env.step(action.item())
        
        step_data = {
            'reward': reward, 
            'mask': bool(not done),
            'state': state,
            'action': action.data,
            'action_logprob': action_log_prob,
            'option': option.data,
            'option_logprob': option_log_prob,
            'term_prob': term_prob,
            'terminate': terminate,
            'env_data': env_data
        }
#         TODO: This should fix the option-state coordination problem but verify
        
        self.current_option = next_option.data
        
        # Push to memory:
        self.memory.push(step_data)
        
        return step_data, next_state, done