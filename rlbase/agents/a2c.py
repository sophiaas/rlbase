import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
# from torch.optim.lr_scheduler import StepLR
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

    def unpack_batch(self, batch):
        # convert pandas series to list 
        # TODO: don't use pandas
        log_probs = batch.log_prob.tolist()
        rewards = batch.reward.tolist()
        values = batch.value.tolist()
        # convert "done" to mask ("not done")
        masks = list(1.0 - batch.done.to_numpy())
        return log_probs, rewards, values, masks

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
        losses = self.compute_loss(batch)
        
        for loss in losses:
            loss.backward()
        
        for opt in self.optimizer.values():
            opt.step()
            
        # empty the replay buffer
        self.replay_buffer.clear()
        
    def compute_loss(self, batch):
        log_probs, rewards, values, masks = self.unpack_batch(batch)
            
        policy_losses = [] # list to save policy (actor) loss
        value_losses = [] # list to save value (critic) loss
              
        returns = self.compute_returns(rewards, masks)
        
        for log_prob, value, r in zip(log_probs, values, returns):
            advantage = r - value.data
            
            # calculate policy (actor) loss
            policy_losses.append(torch.sum(-log_prob * Variable(advantage)))
            
#             rr = torch.Tensor([r]).repeat(value.size())
            target = self.cuda_if_needed(Variable(torch.Tensor([r])))
            
            # calculate value (critic) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, target))
#             value_losses.append(F.smooth_l1_loss(value, target))
        
        # take mean of the losses 
        # ALTERNATIVE: sum
        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()   
        
        loss = [policy_loss, value_loss]
        
        return loss

#         for log_prob, value, r in zip(log_probs, values, returns):
#             reward = r - value.data #[0, 0]  # this also is an issue
#             rr = torch.Tensor([r]).repeat(value.size())  # copies, not merely pointers
#             policy_losses.append(torch.sum(-log_prob * Variable(reward)))
#             target = cuda_if_needed(Variable(rr), self.config)  # basically need r to be [1, num_indices] = [1, shape(value)]
#             value_losses.append(F.smooth_l1_loss(value, target))
