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
        self.data = pd.DataFrame()
        self.checkpoint = {}
        self.episode_data = pd.DataFrame()
        self.data_saved = False
        self.episode_data_saved = False
        
#         #overwrites existing data
#         if os.path.exists(self.logdir+'episode_data.csv'):
#             os.remove(self.logdir+'episode_data.csv')            
#         if os.path.exists(self.logdir+'data.csv'):
#             os.remove(self.logdir+'summary.csv')
        
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
        else:
            os.makedirs(self.logdir)
            os.makedirs(self.checkpointdir)
        
    def save_config(self):
        with open(self.logdir + 'config.p', 'wb') as f:
            pickle.dump(self.config, f)
            
    def load_config(self):
        with open(self.logdir + 'config.p', 'rb') as f:
            config = pickle.load(f)
        return config

    def push(self, data):
        self.data = self.data.append(data, ignore_index=True)

    def push_episode_data(self, episode_data):
        self.episode_data = self.episode_data.append(episode_data, ignore_index=True)

    def save(self):
        self.data.to_pickle(self.logdir+'summary.p')
#         with open(self.logdir+'summary.csv', 'a') as f:
#             if self.data_saved:
#                 self.data.to_csv(f, header=False, index=False)
#             else:
#                 self.data.to_csv(f, header=True, index=False)

    def load(self, name):
        self.data = pd.read_csv(self.logdir+'summary.p')
    
    def save_episode_data(self):
        self.episode_data.to_pickle(self.logdir+'episode_data.p')
#         with open(self.logdir+'episode_data.csv', 'a') as f:
#             if self.episode_data_saved:
#                 self.episode_data.to_csv(f, header=False, index=False)
#             else:
#                 self.episode_data.to_csv(f, header=True, index=False)
        
    def save_checkpoint(self, agent):
        filename = self.checkpointdir+'episode_{}'.format(agent.episode)
        model = {k: v.state_dict() for k,v in agent.model.items()}
        optimizer = {k: v.state_dict() for k,v in agent.optimizer.items()}
        checkpoint = {'model': model, 'optimizer': optimizer}
        torch.save(checkpoint, filename)
        
#     def load_checkpoint(self, filename, checkpoint):
    def plot(self, variable):
        plt.plot(self.data['episode'], self.data[variable])
        plt.xlabel('episode')
        plt.ylabel(variable)
        plt.savefig(os.path.join(self.logdir,'{}.png'.format(variable)))
        plt.clf()
        
#     def plot(self, var1, var2, name):
#         plt.plot(self.data[var1], self.data[var2])
#         plt.xlabel(var1)
#         plt.ylabel(var2)
#         plt.savefig(os.path.join(self.logdir,'{}.png'.format(name)))
#         plt.clf()
