"""
Lightbot
"""

import gym
from gym import spaces
from gym.envs.registration import register
import numpy as np
from .utils import puzzles
import torch
import copy
from collections import OrderedDict


class Lightbot(gym.Env):
    """
    Lightbot environment
    """

    def __init__(self, config):
        """
        Args:
            puzzle_name: Name of the puzzle to run.
                Choices: "zigzag", "cross", "monolith", "stairs"
        """
        self.config = config
        self.board = puzzles.maps[config.puzzle_name]
        self.map = self.board["map"]
        self.extract_map_features()

        self.action_space = spaces.Discrete(5)
        self.raw_observation_space = spaces.Dict(
            {
                "coords": spaces.Tuple(
                    (
                        spaces.Discrete(self.board_size[1]),
                        spaces.Discrete(self.board_size[0]),
                    )
                ),
                "height": spaces.Discrete(self.max_height),
                "direction": spaces.Discrete(4),
                "light_idx": spaces.Discrete(self.num_lights),
                "lights_on": spaces.MultiBinary(self.num_lights),
            }
        )

        # Seed is not needed because the environment is deterministic
        self.set_reward_fn(*[float(x) for x in config.reward_fn.split(",")])
        self.state = self.reset()

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

    def preprocess(self, raw_state):
        """
        state:
        {'lights_on': array([0, 0, 0]), # binary
         'direction': 0,  # in [0, 1, 2, 3]
         'coords': (1, 4),
         'light_idx': -1, # in [-1, 0, 1, 2] # where the largest number is the idx in lights_on
         'height': 0},  # a scalar starting from 0
        """
        lights_on = raw_state["lights_on"]  # (self.num_lights)
        direction = self.index_to_onehot(raw_state["direction"], 4)
        coords = self.coords_to_onehot(raw_state["coords"], self.board_size)

        # because [-1, 0, 1, 2] (light_idx space) --> [0, 1, 2, 3] (onehot space)
        light_idx = self.index_to_onehot(
            raw_state["light_idx"] + 1, self.num_lights + 1
        )

        height = self.index_to_onehot(raw_state["height"], self.max_height + 1)

        state = np.concatenate((lights_on, direction, coords, light_idx, height))

        return state

    def extract_map_features(self):
        board_size = np.shape(self.map)
        num_lights = 0
        height = []
        light_idx = []

        for i in range(board_size[0]):
            for j in range(board_size[1]):
                height.append(self.map[i][j]["h"])

                if self.map[i][j]["t"] == "l":
                    light_idx.append(num_lights)
                    num_lights += 1
                else:
                    light_idx.append(-1)

        x = np.tile(np.arange(board_size[1]), board_size[0])
        y = np.repeat(np.flipud(np.arange(board_size[0])), board_size[1])
        coords = list(zip(x, y))

        self.board_properties = {
            a: {"height": b, "light_idx": c}
            for a, b, c in zip(coords, height, light_idx)
        }
        self.max_height = np.max(height)
        self.board_size = board_size
        self.num_lights = num_lights

    def reset(self):
        if self.config.random_init:
            start_coords = (
                np.random.randint(self.board_size[1]),
                np.random.randint(self.board_size[0]),
            )
            start_direction = np.random.randint(4)
        else:
            start_coords = (self.board["position"]["x"], self.board["position"]["y"])
            start_direction = self.board["direction"]

        self.raw_state = OrderedDict(
            {
                "coords": start_coords,
                "height": self.board_properties[start_coords]["height"],
                "direction": start_direction,
                "light_idx": self.board_properties[start_coords]["light_idx"],
                "lights_on": np.zeros(self.num_lights, dtype=np.int),
            }
        )

        self.state = self.preprocess(self.raw_state)
        self.observation_space = spaces.Discrete(len(self.state))
        self.done = False
        return self.state

    def get_data(self):
        return self.raw_state

    def step(self, action):
        # actions: {0: 'light', 1: 'jump', 2: 'walk', 3: 'turn_r', 4: 'turn_l'}
        # If already terminal, then don't do anything
        if self.done:
            return self.state, 0.0, True, {"state": self.state}, 1
        if type(action) == torch.Tensor:
            action = action.item()
        self.raw_state, self.reward, self.done = self.make_move(
            copy.deepcopy(self.raw_state), action
        )
        self.state = self.preprocess(self.raw_state)
        transition = {
            "next_state": copy.deepcopy(self.state),
            "reward": copy.deepcopy(self.reward),
            "done": copy.deepcopy(self.done),
        }
        #         return transition
        data = self.get_data()
        return (
            copy.deepcopy(self.state),
            copy.deepcopy(self.reward),
            copy.deepcopy(self.done),
            data,
        )

    def make_move(self, state, action):
        state = copy.deepcopy(state)
        # actions: {0: 'light', 1: 'jump', 2: 'walk', 3: 'turn_r', 4: 'turn_l'}
        start_num_lights_on = np.sum(state["lights_on"])

        # light (0) action
        if action == 0:
            if state["light_idx"] != -1 and state["lights_on"][state["light_idx"]] == 0:
                state["lights_on"][state["light_idx"]] = 1

        # jump (1) or walk (2) action
        # directions: {0: 'se', 1: 'ne', 2: 'nw', 3: 'sw'}
        if action == 1 or action == 2:
            if state["direction"] == 0:
                new_coords = (state["coords"][0], state["coords"][1] - 1)  # left
            if state["direction"] == 1:
                new_coords = (state["coords"][0] + 1, state["coords"][1])  # down
            if state["direction"] == 2:
                new_coords = (state["coords"][0], state["coords"][1] + 1)  # right
            if state["direction"] == 3:
                new_coords = (state["coords"][0] - 1, state["coords"][1])  # up
            if new_coords in self.board_properties.keys():
                height_diff = (
                    self.board_properties[(new_coords)]["height"] - state["height"]
                )
                # jump if we can jump exactly 1 up in front of us
                # or jump down. If this condition is not satisfied, then
                # we stay in the same square
                if action == 1 and (height_diff == 1 or height_diff < 0):
                    state["coords"] = new_coords
                    state["height"] = self.board_properties[(new_coords)]["height"]
                    state["light_idx"] = self.board_properties[(new_coords)][
                        "light_idx"
                    ]
                # if the square in front of us is flat, then we move forward one.
                # otherwise nothing happens.
                if action == 2 and height_diff == 0:
                    state["coords"] = new_coords
                    state["light_idx"] = self.board_properties[(new_coords)][
                        "light_idx"
                    ]
                # In all other cases where we stay in the same square.

        # turn right (3) action
        if action == 3:
            state["direction"] = (state["direction"] - 1) % 4

        # turn left (4) action
        if action == 4:
            state["direction"] = (state["direction"] + 1) % 4

        # calculate reward and determine whether game is complete
        end_num_lights_on = np.sum(state["lights_on"])
        lights_diff = end_num_lights_on - start_num_lights_on
        done = True if end_num_lights_on == self.num_lights else False
        reward = self.reward_fn(done, lights_diff)
        return state, reward, done

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

    def get_possible_actions(self, state):
        possible_actions = [3, 4]
        if state["light_idx"] != -1 and state["lights_on"][state["light_idx"]] == 0:
            possible_actions.append(0)
        if state["direction"] == 0:
            new_coords = (state["coords"][0], state["coords"][1] - 1)
        if state["direction"] == 1:
            new_coords = (state["coords"][0] + 1, state["coords"][1])
        if state["direction"] == 2:
            new_coords = (state["coords"][0], state["coords"][1] + 1)
        if state["direction"] == 3:
            new_coords = (state["coords"][0] - 1, state["coords"][1])
        if new_coords in self.board_properties.keys():
            height_diff = (
                self.board_properties[(new_coords)]["height"] - state["height"]
            )
            if height_diff == 1 or height_diff < 0:
                possible_actions.append(1)
            elif height_diff == 0:
                possible_actions.append(2)
        return possible_actions


env_dict = {
    "debug1": "Lightbot-v0",
    "debug2": "Lightbot-v1",
    "0_tutorial": "Lightbot-v2",
    "1_tutorial": "Lightbot-v3",
    "2_tutorial": "Lightbot-v4",
    "stairs": "Lightbot-v5",
    "cross": "Lightbot-v6",
    "monolith": "Lightbot-v7",
    "stairs_2": "Lightbot-v8",
    "little_l": "Lightbot-v9",
    "zigzag": "Lightbot-v10",
    "fractal_cross_0": "Lightbot-v11",
    "fractal_cross_0-1": "Lightbot-v12",
    "fractal_cross_0-2": "Lightbot-v13",
    "fractal_cross_1": "Lightbot-v14",
    "fractal_cross_2": "Lightbot-v15",
}

for k, v in env_dict.items():
    register(id=v, entry_point="lightbot:Lightbot", kwargs={"puzzle_name": k})
