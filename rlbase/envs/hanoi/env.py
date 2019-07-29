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
    id='Hanoi-v0',
    entry_point='hanoi:Hanoi',
)


class Hanoi(gym.Env):

    def set_action_space(self, num_h_actions):
        self.possible_moves = list(itertools.permutations(range(self.num_pegs), 2))
        self.n_actions = len(self.possible_moves)
        self.num_h_actions = num_h_actions
        self.action_space = self.n_actions+self.num_h_actions
#         self.action_space = spaces.Discrete(self.n_actions+self.num_h_actions)
        self.action_dim = self.action_space
        self.action_key = {a: b for a,b in zip(range(self.action_dim), self.possible_moves)}
        self.h_actions = {a: [] for a in range(self.n_actions, self.num_h_actions+self.n_actions)}
        self.open_h_action_index = self.n_actions

    def empty_peg(self, peg):
        if peg == [0] * self.num_disks:
            return True
        else:
            return False
        
    def get_top_disk(self, peg):
        top_disk_idx = np.nonzero(peg)[0][0]
        top_disk = peg[top_disk_idx]    
        return top_disk_idx, top_disk
    
    def dropping_action(self, peg, disk):
        destination = self.raw_state[peg].copy()
        if self.empty_peg(destination):
            if self.verbose:
                print('placing on peg {}'.format(peg))
            dropoff_idx = -1
            return 'valid', dropoff_idx
        else:
            top_disk_idx, top_disk = self.get_top_disk(destination)
            if top_disk < disk:
                return 'invalid', None
            else:
                if self.verbose:
                    print('placing on peg {}'.format(peg))
                dropoff_idx = top_disk_idx-1
                return 'valid', dropoff_idx
    
    def pickup_action(self, peg):
        pickup = self.raw_state[peg]
        if self.empty_peg(pickup):
            return 'invalid', None, None
        top_disk_idx, top_disk = self.get_top_disk(pickup)
        if self.verbose:
            print('picking up from peg {}'.format(peg))
        return 'valid', top_disk_idx, top_disk
    
#     def test_action(self, state, action):
#         move = self.action_key[action]
#         pickup = move[0]
#         dropoff = move[1]
#         status, pickup_idx, pickup_disk = self.pickup_action(pickup)
#         if status == 'invalid':
#             return 'invalid', {}, None
#         else:
#             status, dropoff_idx = self.dropping_action(dropoff, pickup_disk)
#             transition = {'pickup': pickup, 
#                           'pickup_idx': pickup_idx,
#                           'pickup_disk': pickup_disk, 
#                           'dropoff': dropoff, 
#                           'dropoff_idx': dropoff_idx}
#             new_raw_state = copy.deepcopy(self.raw_state)
#             new_raw_state[transition['pickup']][transition['pickup_idx']] = 0
#             destination = new_raw_state[transition['dropoff']].copy()
#             destination[transition['dropoff_idx']] = transition['pickup_disk']
#             new_raw_state[transition['dropoff']] = destination
#             return status, transition, new_raw_state
        
    def make_move(self, raw_state, action, test=False):
        new_raw_state = copy.deepcopy(raw_state)
        done = False
        move = self.action_key[action]
        pickup = move[0]
        dropoff = move[1]
        status, pickup_idx, pickup_disk = self.pickup_action(pickup)
        if status == 'valid':
            status, dropoff_idx = self.dropping_action(dropoff, pickup_disk)
            if status == 'valid':
                new_raw_state[pickup][pickup_idx] = 0
                destination = new_raw_state[dropoff].copy()
                destination[dropoff_idx] = pickup_disk
                new_raw_state[dropoff] = destination
        if new_raw_state in self.raw_goal_states:
            reward = 100
            if self.continual and not test:
                self.set_goal_states(dropoff)
            else:
                done = True
        else:
            reward = -2
#         print('inside {}'.format(reward))
        return copy.deepcopy(new_raw_state), copy.deepcopy(reward), copy.deepcopy(done)
                 
    def step(self, action):
        frame_update = 1
        base_reward = -2
        if self.done:
            self.reward = 0
        if type(action) == torch.Tensor:
            action = action.item()
        if action < self.n_actions:
            possible_actions = self.get_possible_actions(self.raw_state)
            if action in possible_actions:
                self.raw_state, reward, self.done = self.make_move(copy.deepcopy(self.raw_state), action)
#                 print('outside {}'.format(reward))
                self.reward = reward
            else:
                self.reward = base_reward
#             self.raw_state, self.reward, self.done = self.make_move(copy.deepcopy(self.raw_state), action)
        else:
            h_action = self.h_actions[action]
            reward = []
            h_action = self.uncompress_h_action(h_action)
#             done_during_h_action = False
            for a in h_action:
                possible_actions = self.get_possible_actions(self.raw_state)
                if a in possible_actions:
                    self.raw_state, self.reward, self.done = self.make_move(copy.deepcopy(self.raw_state), a)
                else:
                    self.reward = base_reward
#                 self.raw_state, r, self.done = self.make_move(copy.deepcopy(self.raw_state), a)
#                 if done_during_h_action:
#                     r = -2
                reward.append(r)
                if self.done:
                    break
#                     done_during_h_action = True
#             if done_during_h_action:
#                 self.done = True
            frame_update = len(h_action)
            self.reward = reward
        self.frame_count += frame_update
        self.state = copy.deepcopy(self._preprocess(self.raw_state))
        if type(self.reward) == list:
            reward = sum(self.reward)
        else:
            reward = self.reward
        self.total_reward += reward
        
        self.actions_taken.append(action)
        self.raw_states_visited.append(self.raw_state)
        self.states_visited.append(self.state)
        
        if self.frame_count >= self.max_count:
            self.done = True  # death is true here too
            self.death = True
        return copy.deepcopy(self.raw_state), copy.deepcopy(self.reward), copy.deepcopy(self.done), self.death, frame_update
        
#         action_status, transition, new_raw_state  = self.test_action(self.raw_state, action)
#         if self.allow_impossible or action_status == 'valid':
#             self.step_counter += 1
#             if self.step_counter == self.max_count:
#                 self.done = True
#             if action_status == 'valid':
#                 self.raw_state = new_raw_state
#                 self.state = self._preprocess(self.raw_state)
#                 if self.raw_state in self.raw_goal_states:
#                     reward = 100
#                     if self.continual:
#                         self.initial_peg = transition['dropoff']
#                         self.set_goal_states(self.initial_peg)
#                     if not self.continual:
#                         self.done = True
#             return copy.deepcopy(self.raw_state), reward, copy.deepcopy(self.done), {}, 1
#         else:
#             raise error.Error('Unsupported illegal move action')
    
    def get_possible_actions(self, state):
        possible_actions = self.get_possible_primitives(state)
        for h in self.h_actions.keys():
            if len(self.h_actions[h]) > 0:
                if self.check_actions(state, self.h_actions[h]):
                    possible_actions.append(h)
        return possible_actions
    
    def get_possible_primitives(self, state):
        state = copy.deepcopy(state)
        possible_actions = []
        for a in range(self.n_actions):
            new_raw_state, reward, done = self.make_move(state, a, test=True)
            if new_raw_state != state:
                possible_actions.append(a)
        return possible_actions
    
    def check_actions(self, state, actions):
        s = copy.deepcopy(state)
        for a in actions:
            if a in self.get_possible_primitives(s):
                s, reward, done  = self.make_move(s, a, test=True)
            else:
                return False
        return True
    
    def uncompress_h_action(self, h_action):
        primitive = False if np.any([x >= self.n_actions for x in h_action]) else True
        while not primitive:
            primitive_sequence = []
            for x in h_action:
                if x < self.n_actions:
                    primitive_sequence.append(x)
                else: 
                    primitive_sequence += self.h_actions[x]
            h_action = primitive_sequence
            primitive = False if np.any([x >= self.n_actions for x in h_action]) else True
        return h_action
                    
    def set_goal_states(self, initial_peg):
        self.raw_goal_states = []
        for i in range(self.num_pegs):
            if i != initial_peg:
                goal = [[0]*self.num_disks]*self.num_pegs
                goal[i] = list(range(1, self.num_disks+1))
                self.raw_goal_states.append(goal)
    
    def reset_raw_state(self, initial_peg):
        self.raw_state = [[0]*self.num_disks]*self.num_pegs
        self.raw_state[self.initial_peg] = list(range(1,self.num_disks+1))
        
    def update_action_key(self):
        action_key = copy.deepcopy(self.action_key)
        if len(self.action_key) < self.action_space.n:
            for i in range(len(action_key), self.action_space.n):
                action_key[i] = [action_key[x] for x in self.h_actions[i]]
        return action_key
        
    def reset(self):
        self.raw_states_visited = []
        self.states_visited = []
        self.actions_taken = []
        self.done = False
        self.death = False
        self.frame_count = 0
        self.total_reward = 0
        self.step_counter = 0
        
        if not self.fixed_init:
            self.initial_peg = np.random.randint(self.num_pegs)
        self.set_goal_states(self.initial_peg)
        self.reset_raw_state(self.initial_peg)
        self.state = self._preprocess(self.raw_state)
        self.obs_dim = len(self.state)
        self.done = False
        if self.verbose:
            print('Action Key: \n {}'.format(self.action_key))
#         if self.hierarchical:
#             self.action_key = self.update_action_key()
        return copy.deepcopy(self.raw_state)
        
    def _preprocess(self, raw_state):
        processed = copy.deepcopy(raw_state)
        return np.concatenate(processed)

    def render(self):
        print(self.raw_state)

    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def get_num_actions(self):
        return self.action_space
    
    def get_obs_dim(self):
        return self.obs_dim
    
    def set_env_parameters(self, num_disks=4, num_pegs=3, initial_peg=None, 
                           continual=False, max_count=None, allow_impossible=False, 
                           num_h_actions=None, hierarchical_args=None, hierarchical=False, 
                           verbose=False):
        self.continual = continual
        self.num_disks = num_disks
        self.num_pegs = num_pegs
        self.max_count = max_count
        self.hierarchical = hierarchical
        self.hierarchical_args = hierarchical_args
        print('hierarchical args : {}'.format(hierarchical_args))
        if hierarchical:
            self.set_action_space(hierarchical_args['num_h_actions'])
        else:
            self.set_action_space(0)
        if initial_peg:
            self.initial_peg = initial_peg
            self.fixed_init = True
        else:
            self.fixed_init = False
        self.allow_impossible = allow_impossible
        self.verbose = verbose

        if self.verbose:
            print("Hanoi Environment Parameters have been set to:")
            print("\t Number of Disks: {}".format(self.num_disks))
            print("\t Number of Pegs: {}".format(self.num_pegs))
            print("\t Initial Peg: {}".format(self.num_pegs))
