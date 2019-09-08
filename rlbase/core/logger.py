import os
import pickle
import torch
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os

            
class Logger(object):

    def __init__(self, config):
        self.config = config
        self.logdir = config.experiment.base_dir + config.experiment.name + '/'
        self.checkpointdir = self.logdir + 'checkpoints/'
        self.episodedir = self.logdir + 'episodes/'
        self.data = pd.DataFrame()
        self.checkpoint = {}
        self.episode_data = pd.DataFrame()
        self.data_saved = False
        self.episode_data_saved = False
        self.episode = 0
        
        if self.config.experiment.resume:
            assert os.path.exists(self.logdir)
        else:
            self.mkdir()
            
        self.save_config()
        
    def mkdir(self):
        
        if self.config.experiment.debug:
            # overwrite
            if not os.path.exists(self.logdir):
                os.makedirs(self.logdir)
                os.makedirs(self.checkpointdir)
                if self.config.experiment.save_episode_data:
                    os.makedirs(self.episodedir)
        else:
            os.makedirs(self.logdir)
            os.makedirs(self.checkpointdir)
            if self.config.experiment.save_episode_data:
                os.makedirs(self.episodedir)
        
    def save_config(self):
        with open(self.logdir + 'config.p', 'wb') as f:
            pickle.dump(self.config, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    def load_config(self):
        with open(self.logdir + 'config.p', 'rb') as f:
            config = pickle.load(f)
        return config

    def push(self, data):
        self.data = self.data.append(data, ignore_index=True)
        self.episode = int(self.data['episode'].iloc[-1])

    def push_episode_data(self, episode_data):
        self.episode_data = self.episode_data.append(episode_data, ignore_index=True)

    def save(self):
        self.data.to_pickle(self.logdir+'summary.p')

    def load(self, name):
        self.data = pd.read_csv(self.logdir+'summary.p')
    
    def save_episode_data(self):
        self.episode_data.to_pickle(self.episodedir
                                    +'episode_data_{}.p'.format(self.episode))
        self.episode_data.drop(self.episode_data.index, inplace=True)

        
    def save_checkpoint(self, agent):
        filename = self.checkpointdir+'episode_{}'.format(agent.episode)
        policy = agent.policy.state_dict()
        optimizer = agent.optimizer.state_dict()
        checkpoint = {'policy': policy, 'optimizer': optimizer}
        torch.save(checkpoint, filename)
        
    # TODO: load_checkpoint()
        
    def plot(self, variable):
        every_n = self.config.experiment.plot_granularity
        plt.plot(self.data['episode'][::every_n], self.data[variable][::every_n])
        plt.xlabel('episode')
        plt.ylabel(variable)
        plt.savefig(os.path.join(self.logdir,'{}.png'.format(variable)))
        plt.clf()

