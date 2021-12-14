import torch
from torch import nn


class mytp_layer:
    def __init__(self, in_dim, out_dim, activation_function):
        self.invertible = (out_dim == in_dim)
        self.invertible = False

        # weights
        self.weight = torch.empty(out_dim, in_dim, requires_grad=True)
        nn.init.orthogonal_(self.weight)

        self.backweight = self.weight.T.detach().clone().requires_grad_()
        self.backweight.retain_grad()

        # functions
        if activation_function == "leakyrelu":
            self.activation_function = nn.LeakyReLU(0.2)
            self.back_activation_function = nn.LeakyReLU(5)
        elif activation_function == "linear":
            self.activation_function = (lambda x: x)
            self.back_activation_function = (lambda x: x)
        else:
            sys.tracebacklimit = 0
            raise NotImplementedError(f"activation_function : {activation_function} ?")

        # activation
        self.activation = None
        self.linear_activation = None

        # target
        self.target = None

    def forward(self, x, update=True):
        if update:
            self.activation = self.activation_function(x)
            self.linear_activation = self.activation @ self.weight.T
            self.linear_activation.retain_grad()
            return self.linear_activation
        else:
            a = self.activation_function(x)
            h = a @ self.weight.T
            return h

    def backward(self, x):
        a = x @ self.backweight.T
        h = self.back_activation_function(a)
        return h
