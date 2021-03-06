class Memory(object):
    def __init__(
        self,
        features=["action", "state", "logprob", "value", "reward", "mask", "env_data"],
    ):
        self.features = features
        self.clear()

    def clear(self):
        for feature in self.features:
            setattr(self, feature, [])

    def push(self, data):
        for feature, value in data.items():
            self.__dict__[feature].append(value)
