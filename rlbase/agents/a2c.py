import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import random
from core import ReplayBuffer, Memory
from core.utils import cuda_if_needed
from agents import BaseAgent
import numpy as np

class A2CAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.env = self.set_env()
        self.network = self.set_network()
        

class AC_Policy(BaseAgent):
    def __init__(self, config)
#                  num_actions, indim, hdim, nlayers, env, args, epsilon=None, epsilon_greedy=False, num_options=None):
        super(AC_Policy, self).__init__(args)
        self.config = config
        self.env = env

        self.action_head = nn.Linear(hdim, num_actions)

    def forward(self, x, raw_x=None):
        if self.img_obs:
            x = torch.transpose(torch.transpose(x, 1, 3), 2, 3)
            x = self.layers(x)
            x = x.reshape(x.shape[0], -1)
        else:
            for layer in self.layers:
                x = F.relu(layer(x))
        action_scores = self.action_head(x)
        # subtract max log prob
        max_val, max_idx = torch.max(action_scores, 1)
        action_scores = action_scores - torch.max(action_scores[:, max_idx])  # subtract max logit
        probs = torch.exp(action_scores)
        probs = torch.clamp(probs, float(np.finfo(np.float32).eps), 1)  # for numerical instabilities
        if raw_x is not None:
            assert x.size(0) == 1
            if not self.args.img_obs and self.env.allow_impossible == False:
                valid_actions = sorted(self.env.get_possible_actions(raw_x))
                nonvalid_actions = [i for i in range(self.env.get_num_actions()) if i not in valid_actions]
                nonvalid_actions_th = cuda_if_needed(torch.LongTensor(nonvalid_actions), self.args)
    #             zero out non-valid actions
                probs[:, nonvalid_actions_th] = 0
    #             renormalize
                z = torch.sum(probs)
                probs = probs/z
        return probs

    def select_action(self, state, episode):
        action_dist = self.forward(state)
        m = Categorical(action_dist)
        action = m.sample()
        return action.data

    def get_log_prob(self, state, action):
        # not volatile
        action_dist = self.forward(state)
        m = Categorical(action_dist)
        log_prob = m.log_prob(action)
        return log_prob

class AC_Value(BaseAgent):
    def __init__(self, indim, hdim, nlayers, args):
        super(AC_Value, self).__init__(indim, hdim, nlayers, args)
        self.value_head = nn.Linear(hdim, 1)  # AC
        self.img_obs = args.img_obs

    def forward(self, x):
        if self.img_obs:
            x = torch.transpose(torch.transpose(x, 1, 3), 2, 3)
            x = self.layers(x)
            x = x.reshape(x.shape[0], -1)
        else:
            for layer in self.layers:
                x = F.relu(layer(x))
        value = self.value_head(x)
        return value

class AC_Model(object):
    def __init__(self, num_actions, indim, env, args, epsilon=None, epsilon_greedy=False):
        super(AC_Model, self).__init__()
        self.hierarchical = False
        self.args = args
        self.env = env
        self.policy = AC_Policy(num_actions=num_actions, indim=indim, hdim=args.hdim, nlayers=args.nlayers, env=env, args=args)
        self.value_net = AC_Value(indim=indim, hdim=args.hdim, nlayers=args.nlayers, args=args)
        self.model = {'policy': self.policy, 'value_net': self.value_net}
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=args.lr)  # TODO: make this SGD
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=args.lr)  # TODO: make this SGD
        self.optimizer = [self.policy_optimizer, self.value_optimizer]
        self.replay_buffer = Memory()

        print('policy net')
        print(self.policy)
        print('value net')
        print(self.value_net)

    def cuda(self):
        self.policy.cuda()
        self.value_net.cuda()

    def select_action(self, state):
        state = cuda_if_needed(torch.from_numpy(state).float().unsqueeze(0), self.args)
        action = self.policy.select_action(Variable(state, volatile=True), state)  # volatile
        log_prob = self.policy.get_log_prob(Variable(state), Variable(action))  # not volatile
        state_value = self.value_net(Variable(state))  # not volatile
        return action[0], log_prob, state_value

    def compute_returns(self, rewards):
        returns = []
        prev_return = 0
        for r in rewards[::-1]:
            prev_return = r + self.args.gamma * prev_return
            returns.insert(0, prev_return)
        return returns

    def improve(self, lr_mult, optim_epochs=None):
        batch = self.replay_buffer.sample()
        b_lp = batch.logprob
        b_rew = batch.reward
        b_v = batch.value
        b_ret = self.compute_returns(b_rew)
        self.policy_optimizer.lr = self.args.lr * lr_mult
        self.value_optimizer.lr = self.args.lr * lr_mult
        # print 'self.policy_optimizer.lr', self.policy_optimizer.lr
        # print 'self.value_optimizer.lr', self.value_optimizer.lr
        ac_step(b_lp, b_v, b_ret, self.policy_optimizer, self.value_optimizer, self.args)
        self.replay_buffer.clear_buffer()
        # TODO: maybe do something like lr_mult
