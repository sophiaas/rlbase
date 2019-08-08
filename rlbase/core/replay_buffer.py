
class Memory(object):
    
    def __init__(self, features=['actions', 'states', 'logprobs', 'rewards', 'masks']):
        self.features = features
        self.clear()
    
    def clear(self):
        for feature in self.features:
            setattr(self, feature, [])
    
    def push(self, data):
        for feature, value in data.items():
            self.__dict__[feature].append(value)
