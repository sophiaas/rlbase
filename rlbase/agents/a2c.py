import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .base import BaseAgent

"""
Advantage Actor-Critic
"""

class A2C(BaseAgent):
    
    def __init__(self, config):
        super(A2C, self).__init__(config)
        
        self.config = config
        self.set_network_params()
        
        # initialize networks
        self.observation_net = config.network.init_body()
        self.model = config.network.init_heads(self.observation_net)
        self.policy = self.model['policy']
        self.value_net = self.model['value']

        # initialize optimizers
        self.policy_optimizer = config.training.optim(self.policy.parameters(), 
                                    lr=config.training.lr,
                                    weight_decay=config.training.weight_decay)
        self.value_optimizer = config.training.optim(self.value_net.parameters(), 
                                    lr=config.training.lr,
                                    weight_decay=config.training.weight_decay)
        self.optimizer = {'policy': self.policy_optimizer, 
                          'value': self.value_optimizer}
        
        # initialize learning rate schedulers
        self.policy_lr_scheduler = config.training.lr_scheduler( 
                                    self.policy_optimizer, step_size=1, 
                                    gamma=config.training.lr_gamma)
        self.value_lr_scheduler = config.training.lr_scheduler(
                                    self.value_optimizer, step_size=1, 
                                    gamma=config.training.lr_gamma)
        self.lr_scheduler = [self.policy_lr_scheduler, self.value_lr_scheduler]
        
        print('value_net \n {}'.format(self.value_net))
        print('policy \n {}'.format(self.policy))
        
    def set_network_params(self):
        # set network dimensions that are dependent on environment
        self.config.network.body.indim = self.env.observation_space.n
        self.config.network.heads['policy'].outdim = self.env.action_space.n
        self.config.network.heads['value'].outdim = 1
        
    def compute_returns(self, rewards, masks):
        prev_return = 0
        returns = [] # list to save empirical values

        # calculate the discounted empirical value using 
        # rewards returned from the environment
        for i, r in enumerate(reversed(rewards)):
            prev_return = r + self.config.algorithm.gamma * prev_return * masks[i]
            returns.insert(0, prev_return)
        return self.normalize(returns)
        
#     def unpack_batch(self, batch):
#         # convert to tensors
#         states = self.if_cuda(torch.stack(batch.state.tolist()))
#         actions = self.if_cuda(torch.stack(batch.action.to_list()))
#         rewards = self.if_cuda(torch.from_numpy(batch.reward.to_numpy()))
#         masks = self.if_cuda(torch.from_numpy(1.0 - batch.done.to_numpy()))
#         return states, actions, rewards, masks
        
    def unpack_batch(self, batch):
        log_probs = self.if_cuda(torch.stack(batch.log_prob.tolist()))
        rewards = self.if_cuda(torch.tensor(batch.reward.tolist()))
        values = self.if_cuda(torch.stack(batch.value.tolist()))
        masks = self.if_cuda(torch.from_numpy(1.0 - batch.done.to_numpy()))
        
#         log_probs = torch.as_tensor(batch.log_prob.tolist()).requires_grad_()
#         rewards = torch.as_tensor(batch.reward.to_list()).requires_grad_()
#         values = torch.as_tensor(batch.value.tolist()).requires_grad_()
#         masks = torch.as_tensor(list(1.0 - batch.done.to_numpy()))
        
        #V2
#         log_probs = self.tensor(torch.stack(batch.log_prob.tolist()))
#         rewards = self.tensor(torch.from_numpy(batch.reward.to_numpy()))
#         values = self.tensor(torch.stack(batch.value.tolist()))
#         masks = self.tensor(torch.from_numpy(1.0 - batch.done.to_numpy()))
        
        #V1
#         log_probs = self.if_cuda(torch.stack(batch.log_prob.tolist()))
#         rewards = self.if_cuda(torch.from_numpy(batch.reward.to_numpy()))
#         values = self.if_cuda(torch.stack(batch.value.tolist()))
#         masks = self.if_cuda(torch.from_numpy(1.0 - batch.done.to_numpy()))
        return log_probs, rewards, values, masks


        
#     def estimate_advantages(self, rewards, masks, values):
#         # for brevity
#         gamma = self.config.algorithm.gamma
#         tau = self.config.algorithm.gamma
        
# #         tensor_type = type(masks)
#         tensor_shape = values.shape
        
#         # initialize tensors
# #         returns = tensor_type(tensor_shape)
# #         deltas = tensor_type(tensor_shape)
# #         advantages = tensor_type(tensor_shape)
#         returns = torch.zeros(tensor_shape)
#         deltas = torch.zeros(tensor_shape)
#         advantages = torch.zeros(tensor_shape)    
        
#         prev_return = 0
#         prev_value = 0
#         prev_advantage = 0
        
#         # calculate discounted returns and advantages
#         for i in reversed(range(len(rewards))):
#             returns[i] = rewards[i] + gamma * prev_return * masks[i]
#             deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
#             advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

#             prev_return = returns[i, 0]
#             prev_value = values[i, 0]
#             prev_advantage = advantages[i, 0]
# #             prev_return = returns[i]
# #             prev_value = values[i]
# #             prev_advantage = advantages[i]
            
#         advantages = self.normalize(advantages)
#         returns = self.normalize(returns)
                                    
#         return advantages, returns
    
#     def estimate_advantages(self, rewards, masks, values):
#         # for brevity
#         gamma = self.config.algorithm.gamma
#         tau = self.config.algorithm.gamma
        
# #         tensor_type = type(masks)
# #         tensor_shape = values.shape
        
#         # initialize tensors
#         returns = np.zeros(len(rewards))
#         deltas = np.zeros(len(rewards))
#         advantages = np.zeros(len(rewards))
        
#         prev_return = 0
#         prev_value = 0
#         prev_advantage = 0
        
#         # calculate discounted returns and advantages
#         for i in reversed(range(len(rewards))):
#             returns[i] = rewards[i] + gamma * prev_return * masks[i]
#             deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
#             advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

#             prev_return = returns[i]
#             prev_value = values[i]
#             prev_advantage = advantages[i]
        
#         advantages = self.normalize(advantages)
#         returns = self.normalize(returns)
                                    
#         return advantages, returns

    def improve(self):
        # sample a random batch of episodes from replay buffer
        batch = self.replay_buffer.sample()
        
        # step learning rate 
        for sched in self.lr_scheduler:
            sched.step()
        
        # reset gradients
        for opt in self.optimizer.values():
            opt.zero_grad()
        
        # update parameters using computed loss
        self.update_parameters(batch)
            
        # empty the replay buffer
        self.replay_buffer.clear()
        

    def update_parameters(self, batch):
        log_probs, rewards, values, masks = self.unpack_batch(batch)
            
        policy_losses = [] # list to save policy (actor) loss
        value_losses = [] # list to save value (critic) loss
              
        advantages, returns = self.estimate_advantages(rewards, masks, values)
        
        for log_prob, value, advantage, target in zip(log_probs, values, advantages, returns):
            #Make sure no issues with turning into vars and cuda
#             advantage = self.if_cuda(Variable(advantage))
            advantage = torch.as_tensor(advantage)
#             r = self.tensor(r)
#             log_prob = torch.as_tensor(log_prob).requires_grad_()
#             value = torch.as_tensor(value)
#             target = torch.as_tensor(target)
            
#             print('advantage: {}'.format(advantage))
#             print('log_prob: {}'.format(log_prob))
#             print('value: {}'.format(value))
            
#             r = self.if_cuda(Variable(r))
            
            # calculate policy (actor) loss
            policy_losses.append(torch.sum(-log_prob * advantage)) #var advantage
            
            # make sure no issue w change
#             target = r.clone()
#             target = self.tensor(r)
            
            # calculate value (critic) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, target))
        
        # take mean of the losses
        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()   
        
        losses = [policy_loss, value_loss]
        
        for loss in losses:
            loss.backward(retain_graph=True)
        
        for opt in self.optimizer.values():
            opt.step()
        