from net import net
from bp_layer import bp_layer

import torch
import numpy as np
from tqdm import tqdm


class bp_net(net):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_layers(self, in_dim, hid_dim, out_dim, activation_function):
        layers = [None] * self.depth

        # first layer
        layers[0] = bp_layer(in_dim, hid_dim, activation_function)
        # hidden layers
        for i in range(1, self.depth - 1):
            layers[i] = bp_layer(hid_dim, hid_dim, activation_function)
        # last layer
        layers[-1] = bp_layer(hid_dim, out_dim, "linear")

        return layers

    def train(self, train_loader, valid_loader, epochs, lr):
        for e in range(epochs):
            print(f"epochs {e}")

            # monitor
            last_weights = [None] * self.depth
            grad_weights = [0] * self.depth

            # train forward
            for x, y in train_loader:
                y_pred = self.forward(x)

                for d in range(self.depth):
                    last_weights[d] = self.layers[d].weight
                self.update_weights(y, y_pred, lr)
                for d in range(self.depth):
                    grad_weights[d] += torch.norm(last_weights[d] - self.layers[d].weight)

            # predict
            with torch.no_grad():
                print(f"\ttrain : {self.test(train_loader)}")
                print(f"\tvalid : {self.test(valid_loader)}")
                for d in range(self.depth):
                    print(f"\tdW {d}  : {float(grad_weights[d])/len(train_loader)}")

    def update_weights(self, y, y_pred, lr):
        loss = self.loss_function(y_pred, y)
        self.zero_grad()
        loss.backward()
        for d in range(self.depth):
            self.layers[d].weight = (self.layers[d].weight - lr * 1 / len(y) *
                                     self.layers[d].weight.grad).detach().requires_grad_()

    def zero_grad(self):
        for d in range(self.depth):
            if self.layers[d].weight.grad is not None:
                self.layers[d].weight.grad.zero_()
