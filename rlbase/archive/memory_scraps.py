#     def push_save(self):
#         with open(self.dir+'.csv', 'a') as f:
#             if self.saved:
#                 self.memory.to_csv(f, header=False)
#             else:
#                 self.memory.to_csv(f, header=True)
#         self.saved = True
        
#     def append_att(self, att):
#         self.__dict__[att].append(self.a)
                               
            

# class Transition(object):
    
#     def __init__(self, data):
# #         self.features = list(dic.keys())
#         for key, value in data.items():
#             setattr(self, key, value)
    
# #     def clear(self):
# #         for feature in self.features:
# #             setattr(self, feature, [])
    
# #     def append(self, data):
# #         for feature, value in data.items():
# #             self.__dict__[feature].append(value)

# #     def sample(self, batch_size=None):
# #         if batch_size is None:
# #             return self.Transition(*zip(*self.memory))
# #         else:
# #             random_batch = random.sample(self.memory, batch_size)
# #             return self.Transition(*zip(*random_batch))

        
    
# class ReplayBuffer(object):
    
#     def __init__(self, config):
#         self.config = config
#         self.memory = []
# #         self.saved = False
#         self.dir = self.config.experiment.base_dir + 
#                         self.config.experiment.run_data_dir
# #         self.memory_set = False
    
# #     def set_memory(self, data):
        
# #         for key in data.keys():
# #             setattr(self.memory, key, [])
# #         self.memory_set = True
            
#     def push(self, data):
# #         if not self.memory_set:
# #             self.init(data)
#         self.memory.append(Transition(data))
# #         self.memory = self.memory.append(data, ignore_index=True)

#     def sample(self, batch_size=None):
#         if batch_size is None:
#             return self.memory
#         else:
#             return self.memory.sample(batch_size, axis=0)

#     def __len__(self):
#         return len(self.memory)

# #     def clear(self):
# #         self.memory.drop(self.memory.index, inplace=True)
        
# #     def push_save(self):
# #         with open(self.dir+'.csv', 'a') as f:
# #             if self.saved:
# #                 self.memory.to_csv(f, header=False)
# #             else:
# #                 self.memory.to_csv(f, header=True)
# #         self.saved = True
        
#     def append_att(self, att):
#         self.__dict__[att].append(self.a)
                               
            
# # Taken from
# # https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb
# # Transition = namedtuple('Transition', ('state', 'action', 'logprob', 'mask',
# #                                         'next_state', 'reward', 'value'))


# class Memory(object):
#     def __init__(self):
#         self.memory = []
#             self.Transition = namedtuple('Transition', ('state', 'action', 'option', 'terminated', 
#                                                         'logprob', 'mask', 'next_state', 'reward', 'value'))
#         else:
#             self.Transition = namedtuple('Transition', ('state', 'action', 'logprob', 'mask',
#                                         'next_state', 'reward', 'value'))

#     def init_transition(self, data):
        
        
#     def push(self, *args):
#         """Saves a transition."""
#         self.memory.append(self.Transition(*args))

#     def sample(self, batch_size=None):
#         if batch_size is None:
#             return self.Transition(*zip(*self.memory))
#         else:
#             random_batch = random.sample(self.memory, batch_size)
#             return self.Transition(*zip(*random_batch))

#     def append(self, new_memory):
#         self.memory += new_memory.memory

#     def __len__(self):
#         return len(self.memory)

#     def clear_buffer(self):
#         del self.memory[:]

class Memory(object):
    
    def set_features(self, data):
        for key in data.keys():
            setattr(self, key, [])
            
#     def __init__(self, data):
#         for key, value in data.items():
#             setattr(self, key, [])
            
    def clear(self):
        for feature in self.features:
            setattr(self, feature, [])
    
    def append(self, data):
        if self.__dict__ == {}:
            self.set_features(data)
            
        for feature, value in data.items():
            self.__dict__[feature].append(value)

    def sample(self, batch_size=None):
        if batch_size is None:
            return self
        else:
            sample = Memory()
            random_batch = random.sample(self.memory, batch_size)
            sample.append(random_batch)
            return sample


class ReplayBuffer(object):
    
    def __init__(self):
#         self.config = config
#         self.memory = []
#         self.dir = self.config.experiment.base_dir + self.config.experiment.run_data_dir
        self.memory_init = False
    
    def push(self, data):
        if not self.memory_init:
            self.memory = Memory(data)
        self.memory.append(data)

#     def sample(self, batch_size=None):
#         if batch_size is None:
#             return self.memory
#         else:
#             return self.memory.sample(batch_size, axis=0)
        
    def sample(self, batch_size=None):
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return random_batch
#             return Transition(*zip(*random_batch))
        
    def clear(self):
        del self.memory[:]
    
    def __len__(self):
        return len(self.memory)
    
class Memory(object):
    
    def __init__(self, data):
        for key, value in data.items():
            setattr(self, key, [])
            
    def clear(self):
        for feature in self.features:
            setattr(self, feature, [])
    
    def append(self, data):
        for feature, value in data.items():
            self.__dict__[feature].append(value)

    def sample(self, batch_size=None):
        if batch_size is None:
            return self.__
        else:
            random_batch = random.sample(self.memory, batch_size)
            return self.Transition(*zip(*random_batch))


class ReplayBuffer(object):
    
    def __init__(self):
#         self.config = config
        self.memory = []
#         self.dir = self.config.experiment.base_dir + self.config.experiment.run_data_dir
            
    def push(self, data):
        self.memory.append(Transition(data))

#     def sample(self, batch_size=None):
#         if batch_size is None:
#             return self.memory
#         else:
#             return self.memory.sample(batch_size, axis=0)
        
    def sample(self, batch_size=None):
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return random_batch
#             return Transition(*zip(*random_batch))
        
    def clear(self):
        del self.memory[:]
    
    def __len__(self):
        return len(self.memory)
    
class Transition(object):
    
    def __init__(self, data):
        self.__dict__ = data
#         for key, value in data.items():
#             setattr(self, key, value)

class Memory(object):
    
    def __init__(self, data):
        for key, value in data.items():
            setattr(self, key, [])

    
class ReplayBuffer(object):
    
    def __init__(self):
#         self.config = config
        self.memory = []
#         self.dir = self.config.experiment.base_dir + self.config.experiment.run_data_dir
            
    def push(self, data):
        self.memory.append(Transition(data))

#     def sample(self, batch_size=None):
#         if batch_size is None:
#             return self.memory
#         else:
#             return self.memory.sample(batch_size, axis=0)
        
    def sample(self, batch_size=None):
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return random_batch
#             return Transition(*zip(*random_batch))
        
    def clear(self):
        del self.memory[:]
    
    def __len__(self):
        return len(self.memory)
    
import random
class Memory(object):
    def __init__(self):
        self.memory = []
        self.initialized_features = False
#         self.Transition = namedtuple('Transition', ('state', 'action', 'logprob', 'mask',
#                                         'next_state', 'reward', 'value'))
        
    def init_features(self, data):
        self.Transition = namedtuple('Transition', tuple(data.keys()))        

    def push(self, data):
        """Saves a transition."""
        if not self.initialized_features:
            self.init_features(data)
        self.memory.append(self.Transition(data.values()))

    def sample(self, batch_size=None):
        if batch_size is None:
            return self.Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return self.Transition(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)

    def clear_buffer(self):
        del self.memory[:]