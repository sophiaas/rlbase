import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
import math


class Linear3D(nn.Module):
    __constants__ = ["bias"]

    def __init__(self, in_features, in_features_2, out_features):
        """FOR OPTION-CRITIC:
        in_features should be obs_dim or hdim
        in_features_2 should be num_options
        out_features should be hdim or num_actions
        """
        super(Linear3D, self).__init__()
        self.in_features = in_features
        self.in_features_2 = in_features_2
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features_2, in_features, out_features))
        self.bias = Parameter(torch.Tensor(in_features_2, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, dim2_selection=None):
        if len(x.shape) > 1:
            out = torch.zeros((x.shape[0], self.weight.shape[2]))
            for i in range(x.shape[0]):
                out[i] = (
                    torch.matmul(x[i], self.weight[dim2_selection[i]])
                    + self.bias[dim2_selection[i]]
                )
        else:
            W = self.weight[dim2_selection]
            b = self.bias[dim2_selection]
            out = torch.matmul(x, W) + b
        return out

    def extra_repr(self):
        return "in_features_dim1={}, in_features_dim2={}, out_features={}, bias={}".format(
            self.in_features,
            self.in_features_2,
            self.out_features,
            self.bias is not None,
        )
