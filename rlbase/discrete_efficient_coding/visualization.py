import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
# from discrete_efficient_coding.utils import pad_list

def pad_list(l, end_shape=100, pad_element=np.nan):
    new_l = []
    for row in l:
        new_l.append(row + [pad_element]*(end_shape-len(row)))
    return new_l

def plot_grid(batch, title=None, scale=3, img=True, show_colorbar=False, cmap="Greys_r", cols=6):
    N = len(batch)
    cols = cols
    rows = int(math.ceil(np.float(N) / cols))
    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure(figsize=(cols*scale, rows*scale))
    for n, idx in enumerate(batch):
        ax = fig.add_subplot(gs[n])
        if img == True:
            plot = ax.imshow(idx, cmap=cmap)
        else:
            plot = ax.matshow(idx, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        if show_colorbar:
            plt.colorbar(plot, ax=ax)  
    if title is not None:
        fig.suptitle(title, fontsize=30)
        
def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.
    cmap: colormap instance, eg. cm.jet. 
         N: number of colors.     
     Example
     x = resize(arange(100), (5,100))
     djet = cmap_discretize(cm.jet, 5)
     imshow(x, cmap=djet)
     """
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [(indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki]) for i in np.arange(N+1)]
        # Return colormap object.
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)

def get_masks(reward):
    masks = []
    for epoch in reward:
        for i, step in enumerate(epoch):
            if i == len(epoch) - 1:
                masks.append(0)
            else:
                masks.append(1)
    return masks

def estimate_returns(rewards, masks):
    tau = 0.95
    gamma = 1
    returns = np.zeros(rewards.shape[0])  # (B, 1)
    deltas = np.zeros(rewards.shape[0])
    advantages = np.zeros(rewards.shape[0])
    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.shape[0])):
        returns[i] = rewards[i] + gamma * prev_return * masks[i]
#         deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
#         advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]
        prev_return = returns[i]
#         prev_value = values[i]
#         prev_advantage = advantages[i]
#     returns = center(returns)
    return returns

# def estimate_returns(rewards, states, masks, values):
#     tau = 0.95
#     gamma = 1
#     returns = np.zeros(rewards.shape[0])  # (B, 1)
#     deltas = np.zeros(rewards.shape[0])
#     advantages = np.zeros(rewards.shape[0])
#     prev_return = 0
#     prev_value = 0
#     prev_advantage = 0
#     for i in reversed(range(rewards.shape[0])):
#         returns[i] = rewards[i] + gamma * prev_return * masks[i]
#         deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
#         advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]
#         prev_return = returns[i]
#         prev_value = values[i]
#         prev_advantage = advantages[i]
# #     returns = center(returns)
#     return returns, advantages


def unconcat_returns(returns, r):
    returns_by_epoch = []
    prev_end = 0
    for i,row in enumerate(r):
        length = len(row)
        returns_by_epoch.append(list(returns[prev_end: prev_end+length]))
        prev_end += length
    return returns_by_epoch

def get_masks(reward):
    masks = []
    for epoch in reward:
        for i, step in enumerate(epoch):
            if i == len(epoch) - 1:
                masks.append(0)
            else:
                masks.append(1)
    return masks

def plot_trajectories(df, column, discrete=True, cmap='magma', n_intervals=11, n_epochs_per_interval=100, figsize=10):
    breaks = [int(x) for x in np.linspace(0, len(df), n_intervals)]
    prev_end = 0
    vmin = np.min(np.min(df[column]))
    vmax = np.max(np.max(df[column]))
    for i, bound in enumerate(breaks[1:]):
        data = df[column][prev_end:prev_end+n_epochs_per_interval]
        padded = []
        for d in data:
            padded.append(d + [np.nan] * (100 - len(d)))
        if discrete:
            cmap = cmap_discretize(cmap, 5)
        fig = plt.figure(figsize=(figsize,figsize/2))
        plot = plt.imshow(padded, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar()  
        fig.suptitle('epoch {}:{}'.format(prev_end, prev_end+n_epochs_per_interval), fontsize=20)   
        prev_end = bound 

def plot_learning_trajectories(reward, trajectories, n_intervals=11, n_epochs_per_interval=100, figsize=10, returns_cmap='magma', traj_cmap='viridis_r'):
    breaks = [int(x) for x in np.linspace(0, len(reward), n_intervals)]
    print(breaks)
    prev_end = 0
    for i, bound in enumerate(breaks[1:]):
        r = reward[prev_end:prev_end+n_epochs_per_interval]
        t = trajectories[prev_end:prev_end+n_epochs_per_interval]
        masks = get_masks(r)
        r_concat = [item for sublist in r for item in sublist]
        returns = estimate_returns(np.array(r_concat), np.array(masks))
        returns_by_epoch = unconcat_returns(returns, r)
        returns_by_epoch = pad_list(returns_by_epoch)
        end_of_epoch_mask = np.ma.masked_array(returns_by_epoch)
        t_padded = []
        
        for traj in t:
            t_padded.append(traj + [np.nan] * (100 - len(traj)))
        discrete_cmap = cmap_discretize(traj_cmap, 5)

        to_plot = [returns_by_epoch, t_padded]
        cmaps = [returns_cmap, discrete_cmap]
        vmins = [-100, 0]
        vmax = [100, 4]
        
        gs = gridspec.GridSpec(1, 2)
        fig = plt.figure(figsize=(figsize,figsize/2))
        
        for n in range(2):
            ax = fig.add_subplot(gs[n])
            plot = ax.imshow(to_plot[n], cmap=cmaps[n], vmin=vmins[n], vmax=vmax[n])
            plt.colorbar(plot, ax=ax)  
        fig.suptitle('epoch {}:{}'.format(prev_end, prev_end+n_epochs_per_interval), fontsize=20)   
        
        prev_end = bound 
        
def get_running_rewards(mean_rewards):
    running_rewards = []
    for i, r in enumerate(mean_rewards):
        if i == 0:
            rr = r
        else:
            rr = rr * 0.99 + r * 0.01
        running_rewards.append(rr)
    return running_rewards