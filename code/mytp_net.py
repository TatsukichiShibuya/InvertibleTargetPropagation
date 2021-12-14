from net import net
from mytp_layer import mytp_layer

import time
import wandb
import torch
from torch import nn
import numpy as np
from tqdm import tqdm


class mytp_net(net):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.direct_depth = kwargs["direct_depth"]
        assert 1 <= self.direct_depth <= self.depth
        self.MSELoss = nn.MSELoss(reduction="mean")

    def init_layers(self, in_dim, hid_dim, out_dim, activation_function):
        layers = [None] * self.depth

        # first layer
        layers[0] = mytp_layer(in_dim, hid_dim, activation_function, self.device)
        # hidden layers
        for i in range(1, self.depth - 1):
            layers[i] = mytp_layer(hid_dim, hid_dim, activation_function, self.device)
        # last layer
        layers[-1] = mytp_layer(hid_dim, out_dim, "linear", self.device)

        return layers

    def train(self, train_loader, valid_loader, epochs, stepsize, lrb,
              b_epochs, b_sigma, refinement_iter, refinement_type, b_loss, log):
        # train backward network
        for e in range(10):
            # train backward
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                for be in range(b_epochs):
                    self.train_backweights(x, lrb, b_sigma, b_loss)

            # reconstruction loss
            # print(f"epochs {e}: {self.reconstruction_loss_of_dataset(train_loader)}")

        # train forward network
        for e in range(epochs):
            # monitor
            last_weights = [None] * self.depth
            weights_moving = [0] * self.depth
            start_time = time.time()

            # train forward
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                # train backward
                for be in range(b_epochs):
                    self.train_backweights(x, lrb, b_sigma, b_loss)

                # compute target
                self.compute_target(x, y, stepsize, refinement_iter, refinement_type)

                # train forward
                for d in range(self.depth):
                    last_weights[d] = self.layers[d].weight
                self.update_weights(x)
                for d in range(self.depth):
                    weights_moving[d] += torch.norm(last_weights[d] - self.layers[d].weight)

            # predict
            with torch.no_grad():
                train_loss, train_acc = self.test(train_loader)
                valid_loss, valid_acc = self.test(valid_loader)
                rec_loss = self.reconstruction_loss_of_dataset(train_loader)
                if log:
                    log_dict = {"train loss": train_loss,
                                "valid loss": valid_loss,
                                "reconstruction loss": rec_loss}
                    if train_acc is not None:
                        log_dict["train accuracy"] = train_acc
                    if valid_acc is not None:
                        log_dict["valid accuracy"] = valid_acc
                    for d in range(self.depth):
                        log_dict[f"weight moving {d}"] = float(weights_moving[d])
                    log_dict["time"] = time.time() - start_time
                    wandb.log(log_dict)
                else:
                    print(f"epochs {e}")
                    print(f"\ttrain loss     : {train_loss}")
                    print(f"\tvalid loss     : {valid_loss}")
                    if train_acc is not None:
                        print(f"\ttrain acc      : {train_acc}")
                    if valid_acc is not None:
                        print(f"\tvalid acc      : {valid_acc}")
                    print(f"\trec loss       : {rec_loss}")
                    for d in range(self.depth):
                        print(f"\tweight moving {d}: {float(weights_moving[d])}")

    def train_backweights(self, x, lrb, b_sigma, b_loss):
        self.forward(x)
        for d in reversed(range(1, self.depth - self.direct_depth + 1)):
            if b_loss == "inv":
                inverse = torch.linalg.pinv(self.layers[d].weight).to(self.device)
                self.layers[d].backweight = inverse.detach().requires_grad_()
            else:
                if b_loss == "gf":
                    # minimize |q-g(f(q))|^2
                    q = self.layers[d - 1].linear_activation.detach().clone()
                    q += torch.normal(0, b_sigma, size=q.shape, device=self.device)
                    h = self.layers[d].backward(self.layers[d].forward(q, update=False))
                    loss = self.MSELoss(h, q)
                elif b_loss == "fg":
                    # minimize |q-f(g(q))|^2
                    q = self.layers[d].linear_activation.detach().clone()
                    q += torch.normal(0, b_sigma, size=q.shape, device=self.device)
                    h = self.layers[d].forward(self.layers[d].backward(q), update=False)
                    loss = self.MSELoss(h, q)
                elif b_loss == "eye":
                    # minimize |I-WO|^2 + |I-OW|^2
                    eye = torch.eye(self.layers[d].weight.shape[0], device=self.device)
                    loss = torch.norm(eye - self.layers[d].weight @ self.layers[d].backweight)**2
                    loss += torch.norm(eye - self.layers[d].backweight @ self.layers[d].weight)**2

                if self.layers[d].backweight.grad is not None:
                    self.layers[d].backweight.grad.zero_()
                loss.backward()
                self.layers[d].backweight = (self.layers[d].backweight -
                                             lrb * self.layers[d].backweight.grad).detach().requires_grad_()

    def compute_target(self, x, y, stepsize, refinement_iter, refinement_type):
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
            if refinement_type == "gf":
                for i in range(refinement_iter):
                    for d in reversed(range(self.depth - self.direct_depth)):
                        gt = self.layers[d + 1].backward(self.layers[d + 1].target)
                        ft = self.layers[d + 1].forward(self.layers[d].target, update=False)
                        gft = self.layers[d + 1].backward(ft)
                        self.layers[d].target += gt - gft
            elif refinement_type == "fg":
                for i in range(refinement_iter):
                    for d in reversed(range(self.depth - self.direct_depth)):
                        gt = self.layers[d + 1].backward(self.layers[d + 1].target)
                        fgt = self.layers[d + 1].forward(gt, update=False)
                        u = 2 * self.layers[d + 1].target - fgt
                        self.layers[d].target = self.layers[d + 1].backward(u)

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
