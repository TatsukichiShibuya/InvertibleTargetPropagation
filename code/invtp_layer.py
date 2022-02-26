import torch
from torch import nn


class invtp_layer:
    def __init__(self, in_dim, out_dim, activation_function, device):
        # weights
        self.weight = torch.empty(out_dim, in_dim, requires_grad=True, device=device)
        nn.init.orthogonal_(self.weight)

        self.back_weight = self.weight.T.detach().clone().requires_grad_()
        self.back_weight.retain_grad()

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

    def backward(self, h):
        mean, std = torch.mean(self.swx, dim=0), torch.std(self.swx, dim=0)
        s = batch_normalization_inverse(h, mean, std)
        a = self.back_activation_function(s)
        x = s @ self.back_weight.T
        x = batch_normalization(x)
        return x


def batch_normalization(x, mean=None, std=None):
    if mean is None:
        mean = torch.mean(x, dim=0)
    if std is None:
        std = torch.std(x, dim=0)
    return (x - mean) / std


def batch_normalization_inverse(y, mean, std):
    return y * std + mean
