from net import net
from bp_layer import bp_layer

import time
import wandb
import torch
import numpy as np
from tqdm import tqdm


class bp_net(net):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_layers(self, in_dim, hid_dim, out_dim, activation_function):
        layers = [None] * self.depth

        # first layer
        layers[0] = bp_layer(in_dim, hid_dim, activation_function, self.device)
        # hidden layers
        for i in range(1, self.depth - 1):
            layers[i] = bp_layer(hid_dim, hid_dim, activation_function, self.device)
        # last layer
        #layers[-1] = bp_layer(hid_dim, out_dim, "linear", self.device)
        layers[-1] = bp_layer(hid_dim, out_dim, activation_function, self.device)

        return layers

    def train(self, train_loader, valid_loader, epochs, lr, log):
        for e in range(epochs):
            # monitor
            last_weights = [None] * self.depth
            for d in range(self.depth):
                last_weights[d] = self.layers[d].weight
            start_time = time.time()

            # train forward
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.forward(x)

                self.update_weights(y, y_pred, lr)

            end_time = time.time()
            print(f"epochs {e}: {end_time - start_time:.2f}")

            # predict
            with torch.no_grad():
                train_loss, train_acc = self.test(train_loader)
                valid_loss, valid_acc = self.test(valid_loader)

                if log:
                    # results
                    log_dict = {"train loss": train_loss,
                                "valid loss": valid_loss}
                    if train_acc is not None:
                        log_dict["train accuracy"] = train_acc
                    if valid_acc is not None:
                        log_dict["valid accuracy"] = valid_acc
                    log_dict["time"] = end_time - start_time

                    # monitor
                    """
                    for d in range(self.depth):
                        sub = self.MSELoss(self.layers[d].weight, last_weights[d])
                        shape = self.layers[d].weight.shape
                        log_dict[f"weight moving {d}"] = float(sub) / (shape[0] * shape[1])
                    """
                    wandb.log(log_dict)
                else:
                    # results
                    print(f"\ttrain loss     : {train_loss}")
                    print(f"\tvalid loss     : {valid_loss}")
                    if train_acc is not None:
                        print(f"\ttrain acc      : {train_acc}")
                    if valid_acc is not None:
                        print(f"\tvalid acc      : {valid_acc}")

                    # monitor
                    """
                    for d in range(self.depth):
                        sub = self.MSELoss(self.layers[d].weight, last_weights[d])
                        shape = self.layers[d].weight.shape
                        print(f"\tweight moving {d}: {float(sub) / (shape[0] * shape[1])}")
                    for d in range(self.depth):
                        print(f"\tcond {d}: {torch.linalg.cond(self.layers[d].weight)}")
                    """

    def update_weights(self, y, y_pred, lr):
        loss = self.loss_function(y_pred, y)
        batch_size = len(y)
        self.zero_grad()
        loss.backward()
        for d in range(self.depth):
            self.layers[d].weight = (self.layers[d].weight - (lr / batch_size)
                                     * self.layers[d].weight.grad).detach().requires_grad_()

    def zero_grad(self):
        for d in range(self.depth):
            if self.layers[d].weight.grad is not None:
                self.layers[d].weight.grad.zero_()
