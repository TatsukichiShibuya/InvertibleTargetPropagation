import torch
from torch import nn
from utils import batch_normalization, batch_normalization_inverse, get_seed, quantization


class dttp_layer:
    def __init__(self, in_dim, out_dim, activation_function, device):
        # weights
        self.weight = torch.empty(out_dim, in_dim, requires_grad=True, device=device)
        nn.init.orthogonal_(self.weight)

        self.back_weight = torch.empty(in_dim, out_dim, requires_grad=True, device=device)
        nn.init.orthogonal_(self.back_weight)

        # functions
        if activation_function == "linear":
            self.activation_function = (lambda x: x)
            self.back_activation_function = (lambda x: x)
        elif activation_function == "tanh":
            self.activation_function = nn.Tanh()
            self.back_activation_function = nn.Tanh()
        else:
            sys.tracebacklimit = 0
            raise NotImplementedError(f"activation_function : {activation_function} ?")

        # activation
        self.wx = None
        self.swx = None
        self.BNswx = None

        # target
        self.target = None

    def forward(self, x, update=True):
        if update:
            self.wx = x @ self.weight.T
            self.swx = self.activation_function(self.wx)
            # self.BNswx = batch_normalization(self.swx).requires_grad_()
            self.BNswx = self.swx.requires_grad_()
            self.BNswx.retain_grad()
            return self.BNswx
        else:
            a = x @ self.weight.T
            s = self.activation_function(a)
            # h = batch_normalization(s)
            h = s
            return h

    def backward(self, x):
        a = x @ self.back_weight.T
        s = self.back_activation_function(a)
        # h = batch_normalization(s)
        h = s
        return h
