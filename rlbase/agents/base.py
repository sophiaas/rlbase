import copy
import numpy as np
import torch
import gym
from collections import defaultdict

from core.replay_buffer import Memory
from core.logger import Logger

from torch.optim.lr_scheduler import StepLR


"""
Base class for Deep RL agents
"""


class RunningAverage(object):
    def __init__(self):
        super(RunningAverage, self).__init__()
        self.data = {}
        self.alpha = 0.01

    def update_variable(self, key, value):
        self.data[key] = value # overwrite
        if 'running_'+key not in self.data:
            self.data['running_'+key] = value
        else:
            self.data['running_'+key] = (1-self.alpha) * self.data['running_'+key] + self.alpha * value
        return copy.deepcopy(self.data['running_'+key])

    def get_value(self, key):
        if 'running_'+key in self.data:
            return self.data['running_'+key]
        else:
            assert KeyError, "Trying to access {} without giving it an initial value".format('running_'+key)

    def get_last_value(self, key):
        if key in self.data:
            return self.data[key]
        else:
            assert KeyError


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
        # self.episode_steps = 0

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
        print('Episode {} \t avg length: {} \t reward: {}'.format(
            self.episode, 
            round(run_avg.get_value('moves'), 2), 
            round(run_avg.get_value('return'), 2)))
        self.logger.save()
        self.logger.save_checkpoint(self) 
        if self.config.experiment.save_episode_data:
            self.logger.save_episode_data(self.episode)
        self.logger.plot('return')
        self.logger.plot('moves')
  
    def collect_samples(self, run_avg, timestep):
        num_steps = 0
        while num_steps < self.config.training.update_every:
            # print('Starting episode {} at timestep {}'.format(self.episode, timestep + num_steps))
            episode_data, episode_return, episode_length = self.sample_episode(
                self.episode, timestep + num_steps, run_avg)
            num_steps += episode_length
            run_avg.update_variable('return', episode_return)
            run_avg.update_variable('moves', episode_length)
            self.episode += 1

            if self.config.experiment.save_episode_data and self.episode % self.config.experiment.every_n_episodes == 0:
                # print('Pushed episode data at episode: {}'.format(self.episode))
                self.logger.push_episode_data(episode_data)
            if self.episode % self.config.experiment.log_interval == 0:
                self.log(run_avg)

        return num_steps

    def sample_episode(self, episode, step, run_avg):
        episode_return = 0
        self.episode_steps = 0
        episode_data = defaultdict(list, {'episode': int(episode)})
        state = self.env.reset()
        for t in range(self.config.training.update_every):
            state = torch.from_numpy(state).float().to(self.device)
            with torch.no_grad():
                transition, state, done = self.step(state)
            for key, val in transition.items():
                episode_data[key].append(self.convert_data(val))
            episode_return += transition['reward']
            if self.config.experiment.render:
                self.env.render()
            if (step+t+1) % self.config.experiment.num_steps_between_plot == 0:
                summary = {
                    'steps': step+t+1,
                    'return': run_avg.get_value('return'),
                    'moves': run_avg.get_value('moves')
                }
                self.logger.push(summary)
                # print('Pushed summary at step: {}'.format(step+t+1))
            if done:
                break
            self.episode_steps += 1
        episode_length = t+1
        return episode_data, episode_return, episode_length

    def train(self):
        run_avg = RunningAverage()
        timestep = 0
        print('Max Timesteps: {}'.format(self.config.training.max_timesteps))
        while timestep <= self.config.training.max_timesteps:
            num_steps = self.collect_samples(run_avg, timestep)
            timestep += num_steps
            # print('timestep: {} num_steps in this batch: {}'.format(timestep, num_steps))
            self.update()
            self.memory.clear()
        print('Training complete')

    # def train(self):
    #     # TODO: Add handle resumes
    #     timestep = 0
    #     self.episode_steps = 0

    #     run_avg = RunningAverage()
        
    #     episode_reward = 0
    #     episode_data = defaultdict(list, {'episode': int(self.episode)})
    #     state = self.env.reset()

    #     # Iterate through timesteps
    #     print('Max Timesteps: {}'.format(self.config.training.max_timesteps))
    #     for timestep in range(1, self.config.training.max_timesteps+1):

    #         self.episode_steps += 1
    #         state = torch.from_numpy(state).float().to(self.device)

    #         with torch.no_grad():
    #             transition, state, done = self.step(state)
                
    #         for key, val in transition.items():
    #             episode_data[key].append(self.convert_data(val))

    #         episode_reward += transition['reward']

    #         if self.config.experiment.render:
    #             self.env.render()
                    
    #         if self.config.experiment.save_episode_data and self.episode % self.config.experiment.every_n_episodes == 0:
    #             self.logger.push_episode_data(episode_data)
                
    #         if done:
    #             episode_data = defaultdict(list, {'episode': int(self.episode)})
    #             summary = {
    #                 'steps': timestep,
    #                 'return': episode_reward,
    #                 'moves': self.episode_steps
    #             }
    #             self.logger.push(summary)
    #             state = self.env.reset()
                
    #             # Logging
    #             if self.episode % self.config.experiment.log_interval == 0:
                    
    #                 print('Episode {} \t avg length: {} \t reward: {}'.format(self.episode, 
    #                                                                     round(self.episode_steps, 2), 
    #                                                                     round(episode_reward, 2)))

    #                 self.logger.save()
    #                 self.logger.save_checkpoint(self)

    #                 # if self.config.experiment.save_episode_data:
    #                 #     self.logger.save_episode_data(self.episode)

    #                 if self.config.experiment.save_episode_data:
    #                     self.logger.save_episode_data(timestep)


    #                 self.logger.plot('return')
    #                 self.logger.plot('moves')
    #             self.episode += 1
    #             self.episode_steps = 0
    #             episode_reward = 0

    #         if timestep % self.config.training.update_every == 0:
    #             self.update()
    #             self.memory.clear()
            
    #     print('Training complete')
 
    def evaluate(self):
        """TODO: Break out redundant functions from train() and consolidate"""
        returns = []
        num_episodes = 0
        timestep = 0

        # Iterate through episodes            
        for i_episode in range(1, self.config.training.max_episodes+1):
            episode_reward = 0
            out_of_time = False

            episode_data = defaultdict(list, {'episode': int(self.episode)})
            state = self.env.reset()

            # Iterate through steps
            for t in range(1, self.config.training.max_episode_length+1):

                timestep += 1
                state = torch.from_numpy(state).float().to(self.device)
                
                with torch.no_grad():
                    transition, state, done = self.step(state)
                
                for key, val in transition.items():
                    episode_data[key].append(self.convert_data(val))
                
                episode_reward += transition['reward']
                
                if self.config.experiment.render:
                    self.env.render()
                    
                if done:
                    break

                self.memory.clear()
                
                if timestep == self.config.eval.n_eval_steps:
                    out_of_time = True
                    break
            
            self.logger.push_episode_data(episode_data)

            if not out_of_time:
                returns.append(episode_reward)
            else:
                assert timestep == self.config.eval.n_eval_steps
                break  # out of time

            # Logging
            summary = {
            'episode': i_episode,
            'running_rewards': episode_reward,
            'running_moves': t
            }
            self.logger.push(summary)
            self.logger.plot('running_rewards')
            self.logger.plot('running_moves')

            if i_episode % self.config.experiment.log_interval == 0:
                print('Episode {} \t length: {} \t reward: {}'.format(i_episode, t, episode_reward))

        mean_return_across_episodes = np.mean(returns)
        std_return_across_episodes = np.std(returns)

        self.logger.save()

        if self.config.experiment.save_episode_data:
            self.logger.save_episode_data()

        print('Mean Return Across Episodes: {}'.format(mean_return_across_episodes))
        print('Std Return Across Episodes: {}'.format(std_return_across_episodes))

        print('Evaluation complete')
        
    # def evaluate(self):
    #     """TODO: Break out redundant functions from train() and consolidate"""
    #     running_reward = 0
    #     avg_length = 0
    #     timestep = 0
    #     self.episode_steps = 0

    #     # Iterate through episodes            
    #     for i_episode in range(1, self.config.training.max_episodes+1):
    #         episode_reward = 0
    #         episode_data = defaultdict(list, {'episode': int(self.episode)})
    #         state = self.env.reset()
            
    #         # Iterate through steps
    #         for t in range(1, self.config.training.max_episode_length+1):
    #             timestep += 1
    #             self.episode_steps += 1
    #             state = torch.from_numpy(state).float().to(self.device)
                
    #             with torch.no_grad():
    #                 transition, state, done = self.step(state)
                
    #             for key, val in transition.items():
    #                 episode_data[key].append(self.convert_data(val))
                
    #             episode_reward += transition['reward']
    #             running_reward += transition['reward']
                
    #             if self.config.experiment.render:
    #                 self.env.render()
                    
    #             if done:
    #                 break

    #             self.memory.clear()
                
    #             if timestep == self.config.eval.n_eval_steps:
    #                 break
            
    #         self.logger.push_episode_data(episode_data)
                
                
    #         # Logging
    #         if i_episode % self.config.experiment.log_interval == 0:
                
    #             print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, 
    #                                                                 round(self.running_moves, 2), 
    #                                                                 round(self.running_rewards, 2)))
    #             self.logger.save()
                
    #             if self.config.experiment.save_episode_data:
    #                 self.logger.save_episode_data()
                
    #             self.logger.plot('running_rewards')
    #             self.logger.plot('running_moves')
                
    #         avg_length += self.episode_steps
            
    #         if i_episode % 10 == 0:
    #             self.update_running(running_reward/10, avg_length/10)
    #             running_reward = 0
    #             avg_length = 0
    #             self.logger.push(self.get_summary())
                
    #         self.episode_steps = 0
    #         self.episode += 1
            
    #         if timestep == self.config.eval.n_eval_steps:
    #             break
            
    #     print('Evaluation complete')
    
    def step(self):
        return NotImplementedError