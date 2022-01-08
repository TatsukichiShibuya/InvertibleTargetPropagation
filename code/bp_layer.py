import torch
from torch import nn

import sys


class bp_layer:
    def __init__(self, in_dim, out_dim, activation_function, device):
        # weights
        self.weight = torch.empty(out_dim, in_dim, requires_grad=True, device=device)
        nn.init.orthogonal_(self.weight)

        # functions
        if activation_function == "leakyrelu":
            self.activation_function = nn.LeakyReLU(0.2)
        elif activation_function == "relu":
            self.activation_function = nn.LeakyReLU(0)
        elif activation_function == "sigmoid":
            self.activation_function = nn.Sigmoid()
        elif activation_function == "linear":
            self.activation_function = (lambda x: x)
        elif activation_function == "tanh":
            self.activation_function = nn.Tanh()
        else:
            sys.tracebacklimit = 0
            raise NotImplementedError(f"activation_function : {activation_function} ?")

        # activation
        self.linear_activation = None
        self.activation = None

    def forward(self, x, update=True):
        if update:
            self.activation = self.activation_function(x)
            self.linear_activation = self.activation @ self.weight.T
            return self.linear_activation
        else:
            a = self.activation_function(x)
            h = a @ self.weight.T
            return h
