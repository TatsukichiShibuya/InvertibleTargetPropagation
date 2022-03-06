import torch
from torch import nn
from utils import batch_normalization, batch_normalization_inverse, get_seed


class ditp_layer_forward:
    def __init__(self, in_dim, out_dim, activation_function, device, seed):
        # weights
        torch.manual_seed(seed)
        self.weight = torch.empty(out_dim, in_dim, requires_grad=True, device=device)
        nn.init.orthogonal_(self.weight)

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


class ditp_layer_backward:
    def __init__(self, in_dim, hid_dim, out_dim, depth, activation_function, device, seed):
        torch.manual_seed(seed)
        self.depth = depth
        # weights
        self.dims = [0] * (depth + 1)
        self.dims[0], self.dims[-1] = in_dim, out_dim
        for d in range(depth - 1):
            self.dims[d + 1] = hid_dim

        self.weights = [None] * self.depth
        for d in range(self.depth):
            r = 1e-4
            gen = get_seed(d, device)
            self.weights[d] = torch.zeros(size=(self.dims[d + 1], self.dims[d]),
                                          device=device).uniform_(-r, r, generator=gen)

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

    def backward(self, x):
        y = x
        for d in range(self.depth):
            a = y @ self.weights[d].T
            s = self.activation_function(a)
            y = batch_normalization(s)
        return y
