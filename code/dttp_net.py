from net import net
from dttp_layer import dttp_layer

import torch
from torch import nn
import numpy as np
from tqdm import tqdm


class dttp_net(net):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.direct_depth = kwargs["direct_depth"]
        assert 1 <= self.direct_depth <= self.depth
        self.MSELoss = nn.MSELoss(reduction="mean")

    def init_layers(self, in_dim, hid_dim, out_dim, activation_function):
        layers = [None] * self.depth

        # first layer
        layers[0] = dttp_layer(in_dim, hid_dim, activation_function, self.device)
        # hidden layers
        for i in range(1, self.depth - 1):
            layers[i] = dttp_layer(hid_dim, hid_dim, activation_function, self.device)
        # last layer
        layers[-1] = dttp_layer(hid_dim, out_dim, "linear", self.device)

        return layers

    def train(self, train_loader, valid_loader, epochs, stepsize, lrb, b_epochs, sigma):
        # train backward network
        print("-------------------- training BACKWARD network --------------------")
        for e in range(10):
            # train backward
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                for be in range(b_epochs):
                    self.train_backweights(x, lrb, sigma)

            # reconstruction loss
            print(f"epochs {e}: {self.reconstruction_loss_of_dataset(train_loader)}")

        # train forward network
        print("-------------------- training FORWARD network --------------------")
        for e in range(epochs):
            print(f"epochs {e}")

            # monitor
            last_weights = [None] * self.depth
            grad_weights = [0] * self.depth

            # train forward
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                # train backward
                for be in range(b_epochs):
                    self.train_backweights(x, lrb, sigma)

                # compute target
                self.compute_target(x, y, stepsize)

                # train forward
                for d in range(self.depth):
                    last_weights[d] = self.layers[d].weight
                self.update_weights(x)
                for d in range(self.depth):
                    grad_weights[d] += torch.norm(last_weights[d] - self.layers[d].weight)

            # predict
            with torch.no_grad():
                print(f"\ttrain : {self.test(train_loader)}")
                print(f"\tvalid : {self.test(valid_loader)}")
                print(f"\trec   : {self.reconstruction_loss_of_dataset(train_loader)}")
                for d in range(self.depth):
                    print(f"\tdW {d}  : {float(grad_weights[d])/len(train_loader.dataset)}")

    def train_backweights(self, x, lrb, sigma):
        self.forward(x)
        for d in reversed(range(1, self.depth - self.direct_depth + 1)):
            # minimize |q-g(f(q))|^2
            q = self.layers[d - 1].linear_activation.detach().clone()
            q += torch.normal(0, sigma, size=q.shape, device=self.device)
            h = self.layers[d].backward(self.layers[d].forward(q, update=False))
            loss = self.MSELoss(h, q)
            if self.layers[d].backweight.grad is not None:
                self.layers[d].backweight.grad.zero_()
            loss.backward()
            self.layers[d].backweight = (self.layers[d].backweight -
                                         lrb * self.layers[d].backweight.grad).detach().requires_grad_()

    def compute_target(self, x, y, stepsize):
        y_pred = self.forward(x)

        # initialize
        loss = self.loss_function(y_pred, y)
        for d in range(self.depth):
            if self.layers[d].linear_activation.grad is not None:
                self.layers[d].linear_activation.grad.zero_()
        loss.backward(retain_graph=True)
        with torch.no_grad():
            for d in range(self.depth - self.direct_depth, self.depth):
                self.layers[d].target = self.layers[d].linear_activation - \
                    stepsize * self.layers[d].linear_activation.grad
            for d in reversed(range(self.depth - self.direct_depth)):
                self.layers[d].target = self.layers[d + 1].backward(self.layers[d + 1].target)

            # refinement
            for i in range(10):
                for d in reversed(range(self.depth - self.direct_depth)):
                    gt = self.layers[d + 1].backward(self.layers[d + 1].target)
                    ft = self.layers[d + 1].forward(self.layers[d].target, update=False)
                    gft = self.layers[d + 1].backward(ft)
                    self.layers[d].target += gt - gft

    def update_weights(self, x):
        self.forward(x)
        D = self.direct_depth
        global_loss = ((self.layers[-D].target - self.layers[-D].linear_activation)**2).sum(axis=1)
        for d in range(self.depth):
            local_loss = ((self.layers[d].target - self.layers[d].linear_activation)**2).sum(axis=1)
            lr = (global_loss / (local_loss + 1e-12)).reshape(-1, 1)
            n = self.layers[d].activation / \
                (self.layers[d].activation**2).sum(axis=1).reshape(-1, 1)
            grad = (self.layers[d].target - self.layers[d].linear_activation).T @ (n * lr)
            if not (torch.isnan(grad).any() or torch.isinf(grad).any()
                    or torch.isnan(lr).any() or torch.isinf(lr).any()):
                self.layers[d].weight = (self.layers[d].weight + grad).detach().requires_grad_()

    def reconstruction_loss(self, x):
        h1 = self.layers[0].forward(x, update=False)
        h = h1
        for d in range(1, self.depth - self.direct_depth + 1):
            h = self.layers[d].forward(h, update=False)
        for d in reversed(range(1, self.depth - self.direct_depth + 1)):
            h = self.layers[d].backward(h)
        return self.MSELoss(h1, h)

    def reconstruction_loss_of_dataset(self, data_loader):
        rec_loss = 0
        for x, y in data_loader:
            x, y = x.to(self.device), y.to(self.device)
            rec_loss += self.reconstruction_loss(x)
        return rec_loss / len(data_loader.dataset)
