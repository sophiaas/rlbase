import numpy as np
from gym import core, spaces
import gym
from gym.envs.registration import register
import torch


"""
The four rooms environment, adapted from 
https://github.com/jeanharb/option_critic/blob/master/fourrooms/fourrooms.py
"""

class FourRooms(gym.Env):
    
    def __init__(self):
        layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
        self.occupancy = np.array([list(map(lambda c: 1 if c=='w' else 0, line)) for line in layout.splitlines()])

        # From any state the agent can perform one of four actions, up, down, left or right
        self.set_action_space()
        self.observation_space = spaces.Discrete(np.sum(self.occupancy == 0))
        self.obs_dim = self.observation_space.n
        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]
        self.rng = np.random.RandomState(1234)
        self.tostate = {}
        statenum = 0
        for i in range(13):
            for j in range(13):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i,j)] = statenum
                    statenum += 1
        self.tocell = {v:k for k,v in self.tostate.items()}

        self.goal = 62
        self.init_states = list(range(self.observation_space.n))
        self.init_states.remove(self.goal)
        self.frame_count = 0
        self.allow_impossible = True
        self.name = 'fourrooms'

    def empty_around(self, cell):
        avail = []
        for action in range(self.n_actions):
            nextcell = tuple(cell + self.directions[action])
            if not self.occupancy[nextcell]:
                avail.append(nextcell)
        return avail

    def reset(self):
        self.frame_count = 0
        raw_state = self.rng.choice(self.init_states)
        self.state = self.index_to_onehot(raw_state, self.obs_dim)
        self.currentcell = self.tocell[raw_state]
        return self.state

    def index_to_onehot(self, value, dim):
        x = np.zeros(dim)
        x[value] = 1
        return x

    def make_move(self, action):
        nextcell = tuple(self.currentcell + self.directions[action])
        if not self.occupancy[nextcell]:
            self.currentcell = nextcell
            if self.rng.uniform() < 1/3.:
                empty_cells = self.empty_around(self.currentcell)
                self.currentcell = empty_cells[self.rng.randint(len(empty_cells))]

        raw_state = self.tostate[self.currentcell]
        state = self.index_to_onehot(raw_state, self.observation_space.n)
        if raw_state == self.goal:
            done = True
            reward = 100.0
        else: 
            reward = -1.0
            done = False
#         if self.frame_count == self.max_count:
#             done = True
        return state, reward, done
    
        
    def step(self, action):
        """
        The agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect. With probability 2/3, the actions
        cause the agent to move one cell in the corresponding direction, and with probability 1/3,
        the agent moves instead in one of the other three directions, each with 1/9 probability. In
        either case, if the movement would take the agent into a wall then the agent remains in the
        same cell.

        """
        if type(action) == torch.Tensor:
            action = action.item()
            
        state, reward, done = self.make_move(action)
        frame_count = 1
        self.frame_count += frame_count
        
        return state, reward, done, {'coords': self.currentcell}
#         return {'next_state': state, 'reward': reward, 'done': done}
    
    def set_action_space(self):
        # Actions are discrete integer values
        self.n_actions = 4
        self.action_space = spaces.Discrete(self.n_actions)
    
#     def act(self, action):
#         # Dummy method for compatibility
#         return self.step(action)

    def get_frame_count(self):
        return self.frame_count
    
    def get_obs_dim(self):
        return self.obs_dim
    
    def _preprocess(self, state):
        return state
    
    def get_num_actions(self):
        return self.action_space.n

register(
    id='Fourrooms-v0',
    entry_point='fourrooms:Fourrooms',
#     timestep_limit=20000,
    reward_threshold=100,
)