import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
# from torch.optim.lr_scheduler import StepLR
from .base import BaseAgent


class A2C(BaseAgent):
    
    def __init__(self, config):
        super(A2C, self).__init__(config)
        
        self.config = config
        self.set_network_params()

        """FIX THIS"""
        self.observation_net = config.network.body.architecture(config.network.body)
        self.policy = config.network.policy_head.architecture(config.network.policy_head, 
                                                              self.observation_net)
        self.value_net = config.network.value_head.architecture(config.network.value_head, 
                                                           self.observation_net)
        """"""
        self.model = {'policy': self.policy, 'value': self.value_net}

        
        self.policy_optimizer = config.training.optim(self.policy.parameters(), lr=config.training.lr)
        self.value_optimizer = config.training.optim(self.value_net.parameters(), lr=config.training.lr)
        self.optimizer = {'policy': self.policy_optimizer, 'value': self.value_optimizer}
        
        self.policy_lr_scheduler =config.training.lr_scheduler(self.policy_optimizer, step_size=1, 
                                                               gamma=config.training.lr_gamma)
        self.value_lr_scheduler = config.training.lr_scheduler(self.value_optimizer, step_size=1, 
                                                               gamma=config.training.lr_gamma)
        self.lr_scheduler = [self.policy_lr_scheduler, self.value_lr_scheduler]
        
        print('value_net \n {}'.format(self.value_net))
        print('policy \n {}'.format(self.policy))
        
    def set_network_params(self):
        self.config.network.body.indim = self.env.observation_space.n
        self.config.network.policy_head.outdim = self.env.action_space.n
        self.config.network.value_head.outdim = self.env.action_space.n
        
    def compute_returns(self, rewards, masks):
        returns = []
        prev_return = 0
        for i, r in enumerate(reversed(rewards)):
            prev_return = r + self.config.algorithm.gamma * prev_return * masks[i]
            returns.insert(0, prev_return)
        return returns

    def unpack_batch(self, batch):
        log_probs = batch.log_prob.tolist()
        rewards = batch.reward.tolist()
        values = batch.value.tolist()
        masks = list(1.0 - batch.done.to_numpy())
        return log_probs, rewards, values, masks
        
    def compute_loss(self, batch):
        print('batch length: {}'.format(len(batch)))
        log_probs, rewards, values, masks = self.unpack_batch(batch)
        returns = self.compute_returns(rewards, masks)
            
        policy_losses = []
        value_losses = []
               
        for log_prob, value, r in zip(log_probs, values, returns):
            reward = r - value.data
            rr = torch.Tensor([r]).repeat(value.size())
            policy_losses.append(torch.sum(-log_prob * Variable(reward)))
            target = self.cuda_if_needed(Variable(rr))
            value_losses.append(F.smooth_l1_loss(value, target))
        
        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        
        print('policy loss: {}'.format(policy_loss))
            
        loss = [policy_loss, value_loss]
        return loss

#         for log_prob, value, r in zip(log_probs, values, returns):
#             reward = r - value.data #[0, 0]  # this also is an issue
#             rr = torch.Tensor([r]).repeat(value.size())  # copies, not merely pointers
#             policy_losses.append(torch.sum(-log_prob * Variable(reward)))
#             target = cuda_if_needed(Variable(rr), self.config)  # basically need r to be [1, num_indices] = [1, shape(value)]
#             value_losses.append(F.smooth_l1_loss(value, target))
