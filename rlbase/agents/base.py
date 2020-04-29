import copy
import numpy as np
import torch
import gym
from collections import defaultdict

from core.replay_buffer import Memory
from core.logger import Logger
from core.running_average import RunningAverage

from torch.optim.lr_scheduler import StepLR


"""
Base class for Deep RL agents
"""


class BaseAgent(object):
    def __init__(self, config):
        torch.manual_seed(config.experiment.seed)
        self.config = config

        self.env = config.env.init_env()

        self.eps = np.finfo(np.float32).eps.item()
        self.device = config.training.device

        self.memory = Memory()
        self.logger = Logger(config)

        self.model = None
        self.policy = None

        self.episode = 1
        self.episode_steps = 0

    def convert_data(self, x):
        if type(x) == torch.Tensor:
            return x.data.cpu().tolist()
        elif type(x) == bool:
            return int(x)
        elif type(x) == np.ndarray:
            return list(x)
        else:
            return x

    def log(self, run_avg):
        print(
            "Episode {} \t avg length: {} \t reward: {}".format(
                self.episode,
                round(run_avg.get_value("moves"), 2),
                round(run_avg.get_value("return"), 2),
            )
        )
        self.logger.save()
        self.logger.save_checkpoint(self)
        if self.config.experiment.save_episode_data:
            self.logger.save_episode_data(self.episode)
        self.logger.plot("return")
        self.logger.plot("moves")

    def collect_samples(self, run_avg, timestep):
        num_steps = 0
        while num_steps < self.config.training.update_every:
            print(
                "Starting episode {} at timestep {}".format(
                    self.episode, timestep + num_steps
                )
            )
            episode_data, episode_return, episode_length = self.sample_episode(
                self.episode, timestep + num_steps, run_avg
            )
            num_steps += episode_length
            run_avg.update_variable("return", episode_return)
            run_avg.update_variable("moves", episode_length)
            self.episode += 1

            if (
                self.config.experiment.save_episode_data
                and self.episode % self.config.experiment.every_n_episodes == 0
            ):
                print("Pushed episode data at episode: {}".format(self.episode))
                self.logger.push_episode_data(episode_data)
            if self.episode % self.config.experiment.log_interval == 0:
                self.log(run_avg)

        return num_steps

    def sample_episode(self, episode, step, run_avg):
        episode_return = 0
        self.episode_steps = 0
        episode_data = defaultdict(list, {"episode": int(episode)})
        state = self.env.reset()
        for t in range(self.config.training.max_episode_length):
            state = torch.from_numpy(state).float().to(self.device)
            with torch.no_grad():
                transition, state, done = self.step(state)
            for key, val in transition.items():
                episode_data[key].append(self.convert_data(val))
            episode_return += transition["reward"]
            if self.config.experiment.render:
                self.env.render()
            if (step + t + 1) % self.config.experiment.num_steps_between_plot == 0:
                summary = {
                    "steps": step + t + 1,
                    "return": run_avg.get_value("return"),
                    "moves": run_avg.get_value("moves"),
                }
                self.logger.push(summary)

            self.episode_steps += 1
            if self.episode_steps == self.config.training.max_episode_length:
                done = True
            if done:
                break
        episode_length = t + 1
        return episode_data, episode_return, episode_length

    def train(self):
        run_avg = RunningAverage()
        timestep = 0
        print("Max Timesteps: {}".format(self.config.training.max_timesteps))
        while timestep <= self.config.training.max_timesteps:
            num_steps = self.collect_samples(run_avg, timestep)
            timestep += num_steps
            self.update()
            self.memory.clear()
        print("Training complete")

    def evaluate(self):
        timestep = 0
        run_avg = RunningAverage()
        # Iterate through episodes
        while timestep <= self.config.eval.n_eval_steps:
            num_steps = self.collect_samples(run_avg, timestep)
            timestep += num_steps
        print("Evaluation complete")

    def step(self):
        return NotImplementedError

    def update(self):
        return NotImplementedError
