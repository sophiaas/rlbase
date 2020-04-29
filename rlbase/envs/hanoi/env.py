import gym
from gym import error, spaces, utils
from gym.utils import seeding
import math
import itertools
import random
import itertools
import numpy as np
import torch
import copy
import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id="Hanoi-v0", entry_point="hanoi:Hanoi",
)


class Hanoi(gym.Env):
    def __init__(self, config, verbose=False):
        self.config = config
        self.num_pegs = self.config.n_pegs
        self.initial_peg = self.config.initial_peg
        self.num_disks = self.config.n_disks
        self.continual = self.config.continual
        self.set_action_space()
        self.set_reward_fn(*[float(x) for x in config.reward_fn.split(",")])
        self.state = self.reset()
        self.observation_space = spaces.Discrete(len(self.state))
        self.verbose = verbose
        self.name = "hanoi"
        if verbose:
            print("Hanoi Environment Parameters have been set to:")
            print("\t Number of Disks: {}".format(self.num_disks))
            print("\t Number of Pegs: {}".format(self.num_pegs))
            print("\t Initial Peg: {}".format(self.initial_peg))

    def index_to_onehot(self, value, dim):
        x = [0] * dim
        x[value] = 1
        return list(x)

    def reset(self):
        self.done = False
        if self.initial_peg is None:
            self.initial_peg = np.random.randint(self.num_pegs)
        self.set_goal_states(self.initial_peg)
        self.reset_raw_state(self.initial_peg)
        self.state = self._preprocess(self.raw_state)
        return copy.deepcopy(self.state)

    def reset_raw_state(self, initial_peg):
        self.raw_state = [
            self.index_to_onehot(initial_peg, self.num_pegs)
        ] * self.num_disks

    def set_action_space(self):
        self.possible_moves = list(itertools.permutations(range(self.num_pegs), 2))
        self.n_actions = len(self.possible_moves)
        self.action_space = spaces.Discrete(self.n_actions)
        self.action_key = {
            a: b for a, b in zip(range(self.n_actions), self.possible_moves)
        }

    def get_possible_actions(self, state):
        state = copy.deepcopy(state)
        possible_actions = []
        for a in range(self.n_actions):
            new_raw_state, reward, done = self.make_move(state, a, test=True)
            if new_raw_state != state:
                possible_actions.append(a)
        return possible_actions

    def set_goal_states(self, initial_peg):
        self.raw_goal_states = []
        for i in range(self.num_pegs):
            if i != initial_peg:
                goal = [self.index_to_onehot(i, self.num_pegs)] * self.num_disks
                self.raw_goal_states.append(goal)

    def set_reward_fn(self, ifdone, otherwise):
        def reward_fn(done, done_continual):
            if done or done_continual:
                reward = ifdone  # 100
            else:
                reward = otherwise  # -1
            return reward

        self.reward_fn = reward_fn

    def empty_peg(self, peg):
        peg_counts = np.sum(self.raw_state, axis=0)
        if peg_counts[peg] == 0:
            return True
        else:
            return False

    def get_top_disk(self, peg):
        for i, disk in enumerate(self.raw_state):
            if disk == self.index_to_onehot(peg, self.num_pegs):
                return i
        return "empty"

    def make_move(self, raw_state, action, test=False):
        new_raw_state = copy.deepcopy(raw_state)
        done = False
        done_continual = False
        move = self.action_key[action]
        pickup_peg = move[0]
        dropoff_peg = move[1]
        pickup_disk = self.get_top_disk(pickup_peg)
        if pickup_disk != "empty":
            dropoff_disk = self.get_top_disk(dropoff_peg)
            if dropoff_disk == "empty" or dropoff_disk > pickup_disk:
                new_raw_state[pickup_disk] = self.index_to_onehot(
                    dropoff_peg, self.num_pegs
                )

        if new_raw_state in self.raw_goal_states:
            done_continual = True
            done = True
            if self.continual and not test:
                self.initial_peg = dropoff_peg
            else:
                self.initial_peg = None
        reward = self.reward_fn(done, done_continual)
        return copy.deepcopy(new_raw_state), copy.deepcopy(reward), copy.deepcopy(done)

    def step(self, action):
        if type(action) == torch.Tensor:
            action = action.item()
        possible_actions = self.get_possible_actions(self.raw_state)
        if action in possible_actions:
            self.raw_state, reward, self.done = self.make_move(
                copy.deepcopy(self.raw_state), action
            )
        else:
            reward = self.reward_fn(self.done, False)
        self.state = copy.deepcopy(self._preprocess(self.raw_state))
        data = self.get_data()
        return (
            copy.deepcopy(self.state),
            copy.deepcopy(reward),
            copy.deepcopy(self.done),
            data,
        )

    def get_data(self):
        return self.raw_state

    def _preprocess(self, raw_state):
        return np.array(copy.deepcopy(raw_state))

    def render(self):
        print(self.raw_state)
