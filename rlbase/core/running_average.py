import copy


class RunningAverage(object):
    def __init__(self):
        super(RunningAverage, self).__init__()
        self.data = {}
        self.alpha = 0.01

    def update_variable(self, key, value):
        self.data[key] = value  # overwrite
        if "running_" + key not in self.data:
            self.data["running_" + key] = value
        else:
            self.data["running_" + key] = (1 - self.alpha) * self.data[
                "running_" + key
            ] + self.alpha * value
        return copy.deepcopy(self.data["running_" + key])

    def get_value(self, key):
        if "running_" + key in self.data:
            return self.data["running_" + key]
        else:
            assert (
                KeyError
            ), "Trying to access {} without giving it an initial value".format(
                "running_" + key
            )

    def get_last_value(self, key):
        if key in self.data:
            return self.data[key]
        else:
            assert KeyError
