import pandas as pd

    
class ReplayBuffer(object):
    
    def __init__(self, config):
        self.config = config
        self.memory = pd.DataFrame()

    def push(self, data):
        self.memory = self.memory.append(data, ignore_index=True)

    def sample(self, batch_size=None):
        if batch_size is None:
            return self.memory.sample(self.__len__(), axis=0)
        else:
            return self.memory.sample(batch_size, axis=0)

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.drop(self.memory.index, inplace=True)
        