import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.distributions import Categorical
import numpy as np


class OptionCritic(nn.Module):
    def __init__(self, config):
        super(OptionCritic, self).__init__()

        self.config = config
        self.n_options = config.algorithm.n_options
        self.device = self.config.training.device

        self.actor_obs_transform = self.config.network.init_body().to(self.device)
        self.option_actor_obs_transform = self.config.network.init_body().to(
            self.device
        )
        self.critic_obs_transform = self.config.network.init_body().to(self.device)
        self.term_obs_transform = self.config.network.init_body().to(self.device)

        (
            self.actor,
            self.option_actor,
            self.critic,
            self.termination,
        ) = self.config.network.init_heads()

    def forward(self):
        raise NotImplementedError

    def actor_forward(self, state, option):
        x = self.actor_obs_transform(state)
        probs = self.actor(x, option)
        dist = Categorical(probs.to(self.device))
        return dist

    def option_actor_forward(self, state):
        x = self.actor_obs_transform(state)
        probs = self.option_actor(x)
        dist = Categorical(probs)
        return dist

    def term_forward(self, state, option=None):
        x = self.term_obs_transform(state)
        term_probs = self.termination(x)
        if option is not None:
            term_probs = torch.cat(
                [torch.index_select(a, 0, i) for a, i in zip(term_probs, option)]
            )
        return term_probs

    def critic_forward(self, state, option=None):
        x = self.critic_obs_transform(state)
        option_value = torch.squeeze(self.critic(x))
        if option is not None:
            option_value = torch.cat(
                [torch.index_select(a, 0, i) for a, i in zip(option_value, option)]
            )
        return option_value

    def evaluate_action(self, state, action, option):
        dist = self.actor_forward(state.to(self.device), option.to(self.device))
        log_prob = dist.log_prob(action)
        return log_prob

    def evaluate_option(self, state, option):
        dist = self.option_actor_forward(state)
        log_prob = dist.log_prob(option)
        return log_prob

    def option_logprobs_full(self, state):
        full = torch.zeros((state.shape[0], self.n_options)).to(self.device)
        for o in range(self.n_options):
            full[:, o] = self.evaluate_option(state, torch.tensor(o).to(self.device))
        return full

    def act(self, state, option):
        option_dist = self.option_actor_forward(state)
        if option is None:
            option = option_dist.sample()
        option_log_prob = option_dist.log_prob(option)

        action_dist = self.actor_forward(state, option)
        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action)

        term_prob = self.term_forward(state)
        if term_prob[option.data] > torch.tensor(0.5).to(self.device):
            terminate = True
            next_option = option_dist.sample()
        else:
            terminate = False
            next_option = option.clone()

        return (
            action,
            option,
            next_option,
            term_prob,
            terminate,
            action_log_prob,
            option_log_prob,
        )
