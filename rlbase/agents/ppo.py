import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .a2c import A2C

"""
Advantage Actor-Critic (A2C) with Proximal Policy Optimization
"""

class PPO(A2C):
    
    def __init__(self, config):
        super(PPO, self).__init__(config)
    
    def unpack_batch(self, batch):
        # convert to tensors
        states = self.cuda_if_needed(torch.stack(batch.state.tolist()))
        actions = self.cuda_if_needed(torch.stack(batch.action.to_list()))
        rewards = self.cuda_if_needed(torch.from_numpy(batch.reward.to_numpy()))
        masks = self.cuda_if_needed(torch.from_numpy(1.0 - batch.done.to_numpy()))
        return states, actions, rewards, masks

    def policy_step(self, states, actions, returns, advantages, fixed_log_prob):
        # get advantages and log probs
        advantages_var = advantages.view(-1) #var cuda
        log_prob = self.log_prob(states, actions) #var both
        
        # compute policy loss
        ratio = torch.exp(log_prob - fixed_log_prob) #var fixed log prob
        surr1 = ratio * advantages_var
        surr2 = torch.clamp(ratio, 1.0 - self.config.algorithm.clip, 1.0 
                            + self.config.algorithm.clip) * advantages_var
        policy_loss = - torch.min(surr1, surr2).mean()
                                    
        # backprop
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
                                    
        # clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 
                                       self.config.algorithm.clip_norm)
        self.policy_optimizer.step()
    
    def value_step(self, states, returns):
        # empirical values
#         value_target = returns.clone() #var, cuda, no clone
                                    
        # optimize value net predictions
        for i in range(self.config.algorithm.value_iters):
            value_prediction = self.value_net(states) #var
            value_loss = (value_prediction - returns).pow(2).mean() #value_target
                                    
            # backprop
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
    
    def improve(self):
        batch = self.replay_buffer.sample()
        self.update_parameters(batch)
        self.replay_buffer.clear()
    
    def update_parameters(self, batch):
        # for brevity
        minibatch_size = self.config.algorithm.minibatch_size
        optim_epochs = self.config.algorithm.optim_epochs
        
        states, actions, rewards, masks = self.unpack_batch(batch)
        
        with torch.no_grad():
            values = self.value_net(states).data #var
            fixed_log_prob = self.log_prob(states, actions).data #var actions
            
        advantages, returns = self.estimate_advantages(rewards, masks, values)
        
        optim_iter_num = int(np.ceil(states.shape[0] 
                                     / float(minibatch_size))) #??
        
        # POSSIBLE FEATURE: ANNEAL EPOCHS

        for i in range(int(optim_epochs)):
            perm = np.random.permutation(range(states.shape[0]))
            
            # MAKE SURE NOTHING WEIRD HAPPENS WITH VARIABLES AND CUDA
            states = Variable(self.cuda_if_needed(states[perm]))
            actions = Variable(self.cuda_if_needed(actions[perm]))
            returns = Variable(self.cuda_if_needed(returns[perm]))
            advantages = Variable(self.cuda_if_needed(advantages[perm]))
            fixed_log_prob = Variable(self.cuda_if_needed(fixed_log_prob[perm]))

            # iterate through minibatchs
            for j in range(optim_iter_num):
                # get minibatch
                idx = slice(j * minibatch_size, min((j+1)*minibatch_size, 
                                                    states.shape[0]))
                states_batch = states[idx]
                actions_batch = actions[idx]
                advantages_batch = advantages[idx]
                returns_batch = returns[idx]
                fixed_log_prob_batch = fixed_log_prob[idx]

                # anneal learning rate
                for sched in self.lr_scheduler:
                    sched.step()
                    # POSSIBLE FEATURE: ANNEAL ALGORITHM.CLIP BY LR_GAMMA

                # update value network and policy network
                self.value_step(states_batch, returns_batch)
                self.policy_step(states_batch, actions_batch, returns_batch, 
                              advantages_batch, fixed_log_prob_batch)

