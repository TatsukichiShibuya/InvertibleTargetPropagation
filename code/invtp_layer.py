import torch
from torch import nn
from utils import batch_normalization, batch_normalization_inverse, get_seed


class invtp_layer:
    def __init__(self, in_dim, out_dim, activation_function, device):
        # weights
        self.weight = torch.empty(out_dim, in_dim, requires_grad=True, device=device)
        nn.init.orthogonal_(self.weight)

        """
        self.back_weight = self.weight.T.detach().clone().requires_grad_()
        self.back_weight.retain_grad()
        """
        mean, std = self.weight.mean().item(), self.weight.std().item()
        shape = self.weight.T.shape
        generator = get_seed(0, device)
        self.back_weight = torch.zeros(size=shape, device=device).normal_(
            mean, std, generator=generator)

        # functions
        if activation_function == "leakyrelu":
            self.activation_function = nn.LeakyReLU(0.2)
            self.back_activation_function = nn.LeakyReLU(5)
        elif activation_function == "linear":
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
        x = a @ self.back_weight.T
        x = batch_normalization(x)
        return x
