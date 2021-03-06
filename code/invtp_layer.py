import torch
from torch import nn
from utils import batch_normalization, batch_normalization_inverse, get_seed, quantization


class invtp_layer:
    def __init__(self, in_dim, out_dim, activation_function, device, seed, init_params, back_activation_function):
        get_seed(seed, device)
        # weights
        self.weight = torch.empty(out_dim, in_dim, requires_grad=True, device=device)
        nn.init.orthogonal_(self.weight)

        if init_params["dist"] == "uniform":
            shape = self.weight.T.shape
            self.back_weight = torch.zeros(size=shape,
                                           device=device).uniform_(-init_params["range"], init_params["range"],
                                                                   generator=get_seed(seed * 2, device))
        elif init_params["dist"] == "gaussian":
            s = 1e-4
            shape = self.weight.T.shape
            self.back_weight = torch.zeros(size=shape,
                                           device=device).normal_(init_params["mean"], init_params["std"],
                                                                  generator=get_seed(seed * 2, device))
        elif init_params["dist"] == "eye":
            shape = self.weight.T.shape
            self.back_weight = torch.eye(*shape, device=device)
        else:
            sys.tracebacklimit = 0
            raise NotImplementedError(f"activation_function : {activation_function} ?")

        # functions
        if activation_function == "leakyrelu":
            self.activation_function = nn.LeakyReLU(0.2)
        elif activation_function == "linear":
            self.activation_function = (lambda x: x)
        elif activation_function == "tanh":
            self.activation_function = nn.Tanh()
        else:
            sys.tracebacklimit = 0
            raise NotImplementedError(f"activation_function : {activation_function} ?")

        # functions
        if back_activation_function == "leakyrelu":
            self.back_activation_function = nn.LeakyReLU(0.2)
        elif back_activation_function == "linear":
            self.back_activation_function = (lambda x: x)
        elif back_activation_function == "tanh":
            self.back_activation_function = nn.Tanh()
        else:
            sys.tracebacklimit = 0
            raise NotImplementedError(f"back_activation_function : {back_activation_function} ?")

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
            self.BNswx = batch_normalization(self.swx).requires_grad_()
            self.BNswx.retain_grad()
            return self.BNswx
        else:
            a = x @ self.weight.T
            s = self.activation_function(a)
            h = batch_normalization(s)
            return h

    def backward(self, x):
        a = x @ self.back_weight.T
        s = self.back_activation_function(a)
        h = batch_normalization(s)
        return h
