import numpy as np
import torch
import torch.nn as nn
import math
from torch.nn import init
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
def glorot(in_features,out_features):
    init_range = np.sqrt(6.0/(in_features+out_features))
    initial = nn.Parameter(torch.nn.init.uniform_(tensor=torch.Tensor(in_features,out_features), a=-init_range, b=init_range))
    return initial

def kaiming(in_features,out_features): # not used
    initial = nn.Parameter(torch.nn.init.kaiming_uniform_(tensor=torch.Tensor(in_features,out_features), a=math.sqrt(5)))
    return initial

def xavier(in_features,out_features):
    initial = nn.Parameter(torch.nn.init.xavier_uniform_(tensor=torch.Tensor(in_features,out_features), gain=1))
    return initial

def uniform_bias(weight,out_features):
    fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in)
    initial = nn.Parameter(torch.nn.init.uniform_(tensor=torch.Tensor(out_features), a=-bound, b=bound))
    return initial