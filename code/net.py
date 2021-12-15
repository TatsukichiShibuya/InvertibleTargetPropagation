from utils import calc_accuracy

from abc import ABCMeta, abstractmethod
from torch import nn
import torch


class net(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        self.device = kwargs["device"]
        self.depth = kwargs["depth"]
        self.layers = self.init_layers(kwargs["in_dim"],
                                       kwargs["hid_dim"],
                                       kwargs["out_dim"],
                                       kwargs["activation_function"])
        self.loss_function = kwargs["loss_function"]

    def forward(self, x, update=True):
        y = x
        for d in range(self.depth):
            y = self.layers[d].forward(y, update=update)
        return y

    def predict(self, x):
        return self.forward(x, update=False)

    def test(self, data_loader):
        """ return loss, acc """
        pred, label = None, None
        for x, y in data_loader:
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.predict(x)
            pred = y_pred if pred is None else torch.cat([pred, y_pred])
            label = y if label is None else torch.cat([label, y])
        if isinstance(self.loss_function, nn.CrossEntropyLoss):  # classification
            return self.loss_function(pred, label) / len(data_loader.dataset), calc_accuracy(pred, label)
        elif isinstance(self.loss_function, nn.MSELoss):  # regression
            return self.loss_function(pred, label) / len(data_loader.dataset), None

    @abstractmethod
    def init_layers(self, in_dim, hid_dim, out_dim, activation_function):
        raise NotImplementedError
