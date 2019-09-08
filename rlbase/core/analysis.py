import pickle
import pandas as pd
import os


def load_episode_data(model_directory):
    episode_directory = model_directory+'episodes/'
    files = os.listdir(episode_directory)
    df = pd.DataFrame()
    for f in files:
        ep = pd.read_pickle(episode_directory+f)
        df = df.append(ep, ignore_index=True)
    df = df.sort_values(by=['episode'])
    df = df.reset_index(drop=True)
    return df