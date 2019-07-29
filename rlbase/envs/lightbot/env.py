from collections import OrderedDict
import numpy as np
import copy
import pprint
import gym
from gym.envs.registration import register
from gym import spaces
from gym.utils import seeding
from envs.lightbot_env.maps import puzzles
from envs.lightbot_env.maps.map_parsing import extract_map_features
import torch

np.random.seed(0)

env_dict = {
    'debug1': 'Lightbot-v0',
    'debug2': 'Lightbot-v1',
    '0_tutorial': 'Lightbot-v2',
    '1_tutorial': 'Lightbot-v3',
    '2_tutorial': 'Lightbot-v4',
    'stairs': 'Lightbot-v5',
    'cross': 'Lightbot-v6',
    'monolith': 'Lightbot-v7',
    'stairs_2': 'Lightbot-v8',
    'little_l': 'Lightbot-v9',
    'zigzag': 'Lightbot-v10',
    'fractal_cross_0': 'Lightbot-v11',
    'fractal_cross_0-1': 'Lightbot-v12',
    'fractal_cross_0-2': 'Lightbot-v13',
    'fractal_cross_1': 'Lightbot-v14',
    'fractal_cross_2': 'Lightbot-v15'
}

for k, v in env_dict.items():
    register(
        id=v,
        entry_point='lightbot:Lightbot',
        kwargs={
            'puzzle_name': k
        }
    )

# utils
def print_indent(a):
    print('\t' + str(a).replace('\n', '\n\t'))

# class Environment():
#     def reset(self):
#         raise NotImplementedError

#     def render(self):
#         raise NotImplementedError

#     def act(self):
#         raise NotImplementedError

#     def get_frame_count(self):
#         raise NotImplementedError


class Lightbot(gym.Env):
    
    def __init__(self, puzzle_name):
        """
        Args:
            puzzle_name: Name of the puzzle to run.
                Choices: "zigzag", "cross", "monolith", "stairs"
        """
        self.board = puzzles.maps[puzzle_name]
        self.map = self.board['map']
        self.board_size = np.shape(self.map)
        self.board_size, self.board_properties, self.num_lights, self.max_height = \
                                                                extract_map_features(self.map)
        self.observation_space = spaces.Dict({
            "coords": spaces.Tuple((spaces.Discrete(self.board_size[1]),  
                                    spaces.Discrete(self.board_size[0]))), 
            "height": spaces.Discrete(self.max_height), 
            "direction": spaces.Discrete(4), 
            "light_idx": spaces.Discrete(self.num_lights), 
            "lights_on": spaces.MultiBinary(self.num_lights)})
        self.seed()
        self.reward = 0
        self.done = False
        self.reward_fn = None
        
    def set_env_parameters(self, max_count, testing, reward_fn, random_init=False,
                 allow_impossible=True, hierarchical_args=None):
        """
        These are the coords of a board_size = (6,5) board:
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
         (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
         (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),
         (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),
         (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5)]
         NOTE that the FIRST coordinate is in [0,5) and
                   the SECOND coordinate is in [0, 6)!
            this means that the first coordinate goes from top to bottom
            and the second coordinate goes from left to right

        let i index up/down (first coord)
        let j index left/right (second coord)
        board_size = (second_coord, first_coord)

        reward_fn is a string: "100,10,-1,-1"
        """
        self.random_init = random_init
        if hierarchical_args is not None:
            self.set_action_space(hierarchical_args['num_h_actions'])
            self.hierarchical_args = hierarchical_args
        else:
            self.set_action_space(0)
        self.set_reward_fn(*[float(x) for x in reward_fn.split(',')])
        self.max_count = max_count
        self.allow_impossible = allow_impossible
        self.testing = testing

        self.board_size = self.get_board_size()
        self.board_properties = self.get_board_properties()
        self.num_lights = self.get_num_lights()
        self.max_height = self.get_max_height()
        self.num_directions = 4

        self.action_space = self.get_action_space()
        self.obs_dim = self.num_lights + \
            self.num_directions + \
            np.prod(self.board_size) + \
            self.num_lights + 1 + \
            self.max_height + 1
        self.frame_count = 0
        self.total_reward = 0

        self.raw_states_visited = []
        self.states_visited = []
        self.actions_taken = []
        self.done = False
        self.death = False
        
    def set_action_space(self, num_h_actions):
        self.n_actions = 5
        self.num_h_actions = num_h_actions
        self.action_space = spaces.Discrete(self.n_actions+self.num_h_actions)
        self.h_actions = {a:[] for a in range(self.n_actions, self.num_h_actions+self.n_actions)}
        self.open_h_action_index = self.n_actions
        
    def reset(self):
        self.raw_states_visited = []
        self.states_visited = []
        self.actions_taken = []
        self.done = False
        self.death = False

        # reset env
        self.frame_count = 0
        self.total_reward = 0
        if self.random_init:
            start_coords = (np.random.randint(self.board_size[1]), np.random.randint(self.board_size[0]))
            start_direction = np.random.randint(4)
        else:
            start_coords = (self.board['position']['x'], self.board['position']['y'])
            start_direction = self.board['direction']
        self.raw_state = OrderedDict({"coords": start_coords,
                                  "height": self.board_properties[start_coords]['height'],
                                  "direction": start_direction,
                                  "light_idx": self.board_properties[start_coords]['light_idx'],
                                  "lights_on": np.zeros(self.num_lights, dtype=np.int)})
        self.state = copy.deepcopy(self._preprocess(self.raw_state))

        # log
        self.raw_states_visited.append(self.raw_state)
        self.states_visited.append(self.state)
        return self.raw_state

    def get_possible_actions(self, state):
        possible_actions = self.get_possible_primitives(state)
        for h in self.h_actions.keys():
            if len(self.h_actions[h]) > 0:
                if self.check_actions(state, self.h_actions[h]):
                    possible_actions.append(h)
        return possible_actions
    
    def get_possible_primitives(self, state):
        possible_actions = [3, 4]
        if state['light_idx'] != -1 and state['lights_on'][state['light_idx']] == 0:
            possible_actions.append(0)
        if state['direction'] == 0:
            new_coords = (state["coords"][0], state["coords"][1] - 1)
        if state['direction'] == 1:
            new_coords = (state["coords"][0] + 1, state["coords"][1])
        if state['direction'] == 2:
            new_coords = (state["coords"][0], state["coords"][1] + 1)
        if state['direction'] == 3:
            new_coords = (state["coords"][0] - 1, state["coords"][1])
        if new_coords in self.board_properties.keys():
            height_diff = self.board_properties[(new_coords)]["height"] - state["height"]
            if height_diff == 1 or height_diff < 0:
                possible_actions.append(1)
            elif height_diff == 0:
                possible_actions.append(2)   
        return possible_actions
    
    def check_actions(self, state, actions):
        s = state.copy()
        for a in actions:
            if a in self.get_possible_primitives(s):
                s, _, _ = self._make_move(s, a)
            else:
                return False
        return True
    
    def set_reward_fn(self, ifdone, ifmorelight, iflesslight, otherwise):
        def reward_fn(done, lights_diff):
            if done == True:
                reward = ifdone  # 100
            elif lights_diff == 1:
                reward = ifmorelight  # 10
            elif lights_diff == -1:
                reward = iflesslight  # -1
            else:
                reward = otherwise  # -1
            return reward
        self.reward_fn = reward_fn

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
    
    def uncompress_h_action(self, h_action):
        primitive = False if np.any([x >= 5 for x in h_action]) else True
        while not primitive:
            primitive_sequence = []
            for x in h_action:
                if x < 5:
                    primitive_sequence.append(x)
                else: 
                    primitive_sequence += self.h_actions[x]
            h_action = primitive_sequence
            primitive = False if np.any([x >= 5 for x in h_action]) else True
        return h_action

    def _make_move(self, state, action):
        state = copy.deepcopy(state)
        # actions: {0: 'light', 1: 'jump', 2: 'walk', 3: 'turn_r', 4: 'turn_l'}
        start_num_lights_on = np.sum(state['lights_on'])
        

        #light action
        if action == 0:
            if state['light_idx'] != -1 and state['lights_on'][state['light_idx']]==0:
                state['lights_on'][state['light_idx']] = 1

        # jump (1) or walk (2) action
            # directions: {0: 'se', 1: 'ne', 2: 'nw', 3: 'sw'}
        if action == 1 or action == 2:
            if state['direction'] == 0:
                new_coords = (state["coords"][0], state["coords"][1] - 1)  # left
            if state['direction'] == 1:
                new_coords = (state["coords"][0] + 1, state["coords"][1])  # down
            if state['direction'] == 2:
                new_coords = (state["coords"][0], state["coords"][1] + 1)  # right
            if state['direction'] == 3:
                new_coords = (state["coords"][0] - 1, state["coords"][1])  # up
            if new_coords in self.board_properties.keys():
                height_diff = self.board_properties[(new_coords)]["height"] - state["height"]
                # jump if we can jump exactly 1 up in front of us
                # or jump down. If this condition is not satisfied, then
                # we stay in the same square
                if action == 1 and (height_diff == 1 or height_diff < 0):
                    state["coords"] = new_coords
                    state["height"] = self.board_properties[(new_coords)]["height"]
                    state["light_idx"] = self.board_properties[(new_coords)]["light_idx"]
                # if the square in front of us is flat, then we move forward one.
                # otherwise nothing happens.
                if action == 2 and height_diff == 0:
                    state["coords"] = new_coords
                    state["light_idx"] = self.board_properties[(new_coords)]["light_idx"]
                # In all other cases where we stay in the same square.

        # turn_r action
        if action == 3:
            state['direction'] = (state['direction'] - 1) % 4

        # turn_l action
        if action == 4:
            state['direction'] = (state['direction'] + 1) % 4

        # calculate reward and determine whether game is complete
        end_num_lights_on = np.sum(state['lights_on'])
        lights_diff = end_num_lights_on - start_num_lights_on
        done = True if end_num_lights_on == self.num_lights else False
        reward = self.reward_fn(done, lights_diff)
        return state, reward, done

    def step(self, action):
        frame_update = 1
        # actions: {0: 'light', 1: 'jump', 2: 'walk', 3: 'turn_r', 4: 'turn_l'}
        # If already terminal, then don't do anything
        if self.done:
            self.reward = 0
        if type(action) == torch.Tensor:
            action = action.item()
        if action < 5:
            possible_actions = self.get_possible_actions(self.raw_state)
            if action in possible_actions:
                self.raw_state, self.reward, self.done = self._make_move(copy.deepcopy(self.raw_state), action)
            else:
                self.reward = -1
        else:
            h_action = self.h_actions[action]
            reward = []
            h_action = self.uncompress_h_action(h_action)
#             done_during_h_action = False
            for a in h_action:
                possible_actions = self.get_possible_actions(self.raw_state)
                if action in possible_actions:
                    self.raw_state, r, self.done = self._make_move(copy.deepcopy(self.raw_state), a)
                else:
                    r = -1
#                 if done_during_h_action:
#                     r = -1
                reward.append(r)
                if self.done:
                    self.reward = reward
                    frame_update = len(h_action)
#                     done_during_h_action = True 
                    break
            self.reward = reward
            
#         self.raw_state, self.reward, self.done, frame_update = copy.deepcopy(self.update(action))
        self.frame_count += frame_update
        self.state = copy.deepcopy(self._preprocess(self.raw_state))
        if type(self.reward) == list:
            reward = sum(self.reward)
        else:
            reward = self.reward
        self.total_reward += reward

        # log
        self.actions_taken.append(action)
        self.raw_states_visited.append(self.raw_state)
        self.states_visited.append(self.state)

        if self.frame_count >= self.max_count:
            self.done = True  # death is true here too
            self.death = True
        return copy.deepcopy(self.raw_state), copy.deepcopy(self.reward), copy.deepcopy(self.done), self.death, frame_update

    def act(self, action):
        return self.step(action)

    def index_to_onehot(self, value, dim):
        x = np.zeros(dim)
        x[value] = 1
        return x

    def coords_to_onehot(self, coords, boundaries):
        dim = np.prod(boundaries)
        w, h = boundaries
        x = np.zeros(dim)
        idx = coords[0] * w + coords[1]  # this should be w !!!!!!
        x[idx] = 1
        return x

    def _preprocess(self, raw_state):
        """
        {'state':
          {'lights_on': array([0, 0, 0]), # binary
           'direction': 0,  # in [0, 1, 2, 3]
           'coords': (1, 4),
           'light_idx': -1, # in [-1, 0, 1, 2] # where the largest number is the idx in lights_on
           'height': 0},  # a scalar starting from 0
          'reward': 0,  # +10 if you increased the number of lights by 1, -10 if decreased by 1, 0 otherwise
          'complete': False}  # in {True, False}
        """
        lights_on = raw_state['lights_on']  # (self.num_lights)
        direction = self.index_to_onehot(raw_state['direction'], 4)
        coords = self.coords_to_onehot(
            raw_state['coords'], self.board_size)
        # because [-1, 0, 1, 2] (light_idx space) --> [0, 1, 2, 3] (onehot space)
        light_idx = self.index_to_onehot(
            raw_state['light_idx'] + 1, self.num_lights + 1)
        height = self.index_to_onehot(raw_state['height'], self.max_height + 1)
        state = np.concatenate(
            (lights_on, direction, coords, light_idx, height))
        return state

    # NOTE just a dummy method
    def get_frame_count(self):
        return self.frame_count

    def get_num_actions(self):
        return self.action_space

    def get_board_size(self):
        return copy.deepcopy(self.board_size)

    def get_board_properties(self):
        return copy.deepcopy(self.board_properties)

    def get_num_lights(self):
        return copy.deepcopy(self.num_lights)

    def get_max_height(self):
        return copy.deepcopy(self.max_height)

    def get_action_space(self):
        return copy.deepcopy(self.action_space.n)

    def get_obs_dim(self):
        return self.obs_dim

    def render(self, initial=False):
        """
            directions:
                0: <, 1: v, 2: >, 3: ^

            Cells are coded as 'abcd'
                a represents the agent's direction:
                    '<', 'v', '>', '^', or ' ' (if no agent)
                b represents the height of the cell
                c is * if contains light else ' '
                d is ! if light on, ' ' if light off

            actions:
                {0: 'light', 1: 'jump', 2: 'walk', 3: 'right', 4: 'left'}
        """
        actions = {0: 'light', 1: 'jump', 2: 'walk', 3: 'right', 4: 'left'}
        directions = ['<', 'v', '>', '^']
        w, h = self.board_size
        print('#'*80)
        if initial:
            print('STARTING RAW STATE\n\t', pprint.pprint(self.raw_state))
        else:
            print('RAW STATE\n\t', pprint.pprint(OrderedDict(self.raw_states_visited[-2])))
            print('STATE', print_indent(self.states_visited[-2]))
            print('\nACTION:\t{} ({})'.format(self.actions_taken[-1], actions[self.actions_taken[-1]]))
            print('\nNEXT RAW STATE\n\t', pprint.pprint(self.raw_states_visited[-1]))
            print('NEXT STATE', print_indent(self.states_visited[-1]))
            print('\nREWARD\t{}'.format(self.reward))
            # print '\nTOTAL REWARD \t{}'.format(self.env.get_total_reward())
            print('\nTOTAL REWARD \t{}'.format(self.total_reward))
            print('\nDONE\t{}'.format(self.done))
            print('\nDEATH\t{}'.format(self.death))
        board = [[None for j in range(w)] for i in range(h)]
        for row in range(h):
            for col in range(w):
                cell = self.board_properties[(row, col)]
                a = directions[self.raw_state['direction']] if self.raw_state['coords'] == (row, col) else ' '
                b = cell['height']
                c = '*' if cell['light_idx'] > -1 else ' '
                d = '!' if cell['light_idx'] > -1 and self.raw_state['lights_on'][cell['light_idx']] > 0 else ' '
                descr = '{}{}{}{}'.format(a,b,c,d)
                board[row][col] = descr
        print('\nBOARD (directions: 0: < || 1: v || 2: > || 3: ^)\n')
        print_indent(np.array(board))
        print('#'*80)

        if self.death:
            print('%'*80 + '\n' + '%'*16 + \
                ' EPISODE TERMINATED BECAUSE MAX LENGTH REACHED ' + \
                '%'*17 + '\n' + '%'*80)  # CHANGED

        assert self.raw_state == self.raw_states_visited[-1]
