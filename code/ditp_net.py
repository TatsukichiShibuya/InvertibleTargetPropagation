from net import net
from ditp_layer import ditp_layer_forward, ditp_layer_backward
from utils import calc_angle, batch_normalization

import sys
import time
import wandb
import torch
from torch import nn
import numpy as np


class ditp_net(net):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.direct_depth = kwargs["direct_depth"]
        assert 1 <= self.direct_depth <= self.depth
        self.layers_backward = self.init_layers_backward(kwargs["in_dim"],
                                                         kwargs["hid_dim"],
                                                         kwargs["out_dim"],
                                                         kwargs["activation_function"])

    def init_layers(self, in_dim, hid_dim, out_dim, activation_function):
        layers_forward = [None] * self.depth

        # dims
        self.dims = [0] * (self.depth + 1)
        self.dims[0], self.dims[-1] = in_dim, out_dim
        for d in range(self.depth - 1):
            self.dims[d + 1] = hid_dim

        # forward
        for d in range(self.depth):
            layers_forward[d] = ditp_layer_forward(self.dims[d], self.dims[d + 1],
                                                   activation_function, self.device, (d + 1) * 11)
        return layers_forward

    def init_layers_backward(self, in_dim, hid_dim, out_dim, activation_function):
        layers_backward = [None] * (self.depth - self.direct_depth)

        # dims
        self.dims = [0] * (self.depth + 1)
        self.dims[0], self.dims[-1] = in_dim, out_dim
        for d in range(self.depth - 1):
            self.dims[d + 1] = hid_dim

        # backward
        D = self.depth - self.direct_depth
        for d in range(D):
            layers_backward[d] = ditp_layer_backward(self.dims[D + 1], hid_dim,  self.dims[D - d], 2,
                                                     activation_function, self.device, (d + 1) * 13)

        return layers_backward

    def train(self, train_loader, valid_loader, epochs, stepsize, lr, log):
        # reconstruction loss
        rec_loss = self.reconstruction_loss_of_dataset(train_loader)
        if torch.isnan(rec_loss).any():
            print("ERROR: rec loss diverged")
            sys.exit(1)
        print(f"initial rec loss: {rec_loss}")

        # train forward network
        for e in range(epochs):
            torch.cuda.empty_cache()
            target_ratio_sum = [0] * (self.depth - self.direct_depth)
            target_angle_sum = [0] * (self.depth - self.direct_depth)
            bp_angle_sum = [0] * self.depth

            monitor_time = 0
            start_time = time.time()

            # train backward
            self.train_back_weights(e)

            rec_loss = self.reconstruction_loss_of_dataset(train_loader)
            if torch.isnan(rec_loss).any():
                print("ERROR: rec loss diverged")
                sys.exit(1)
            print(f"before epochs {e}:\n\trec loss       : {rec_loss}")

            # train forward
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                # compute target
                self.compute_target(x, y, stepsize)

                ###### monitor start ######
                monitor_start_time = time.time()
                with torch.no_grad():
                    D = self.depth - self.direct_depth
                    for d1 in range(D):
                        t = self.layers[d1].target
                        for d2 in range(d1 + 1, D + 1):
                            t = self.layers[d2].forward(t, update=False)
                        v1 = self.layers[D].BNswx - t
                        v2 = self.layers[D].BNswx - self.layers[D].target
                        nonzero = torch.norm(v2, dim=1) > 1e-6
                        target_ratio = torch.norm(v1[nonzero], dim=1) / \
                            torch.norm(v2[nonzero], dim=1)
                        target_ratio_sum[d1] = target_ratio_sum[d1] + target_ratio.sum()
                        target_angle = calc_angle(v1[nonzero], v2[nonzero])
                        target_angle_sum[d1] = target_angle_sum[d1] + target_angle.sum()
                    monitor_end_time = time.time()
                    monitor_time = monitor_time + monitor_end_time - monitor_start_time
                ###### monitor end ######

                ###### monitor start ######
                monitor_start_time = time.time()
                y_pred = self.forward(x)
                loss = self.loss_function(y_pred, y)
                for d in range(self.depth):
                    if self.layers[d].weight.grad is not None:
                        self.layers[d].weight.grad.zero_()
                loss.backward()
                bp_grad = [None] * self.depth
                for d in range(self.depth):
                    bp_grad[d] = self.layers[d].weight.grad.clone()
                monitor_end_time = time.time()
                monitor_time = monitor_time + monitor_end_time - monitor_start_time
                ###### monitor end ######

                # train forward
                tp_grad = self.update_weights(x, lr)
                for d in range(self.depth):
                    bp_angle_sum[d] += calc_angle(tp_grad[d].reshape((1, -1)),
                                                  bp_grad[d].reshape((1, -1))).sum()

            end_time = time.time()
            print(f"epochs {e}: {end_time - start_time - monitor_time:.2f}, {monitor_time:.2f}")

            # predict
            with torch.no_grad():
                train_loss, train_acc = self.test(train_loader)
                valid_loss, valid_acc = self.test(valid_loader)
                rec_loss = self.reconstruction_loss_of_dataset(train_loader)
                if torch.isnan(rec_loss).any():
                    print("ERROR: rec loss diverged")
                    sys.exit(1)

                if log:
                    # results
                    log_dict = {"train loss": train_loss,
                                "valid loss": valid_loss,
                                "reconstruction loss": rec_loss}
                    if train_acc is not None:
                        log_dict["train accuracy"] = train_acc
                    if valid_acc is not None:
                        log_dict["valid accuracy"] = valid_acc
                    log_dict["time"] = end_time - start_time - monitor_time

                    # monitor
                    datasize = len(train_loader.dataset)
                    for d in range(self.depth - self.direct_depth):
                        log_dict[f"target ratio {d}"] = target_ratio_sum[d].item() / datasize
                        log_dict[f"target angle {d}"] = target_angle_sum[d].item() / datasize

                    for d in range(self.depth):
                        log_dict[f"BP angle {d}"] = bp_angle_sum[d].item() / len(train_loader)

                    wandb.log(log_dict)

                else:
                    # results
                    print(f"\ttrain loss     : {train_loss}")
                    print(f"\tvalid loss     : {valid_loss}")
                    if train_acc is not None:
                        print(f"\ttrain acc      : {train_acc}")
                    if valid_acc is not None:
                        print(f"\tvalid acc      : {valid_acc}")
                    print(f"\trec loss       : {rec_loss}")

                    # monitor
                    datasize = len(train_loader.dataset)
                    for d in range(self.depth - self.direct_depth):
                        print(f"\ttarget ratio {d}: {target_ratio_sum[d].item() / datasize}")
                        print(f"\ttarget angle {d}: {target_angle_sum[d].item() / datasize}")

                    for d in range(self.depth):
                        print(f"\tBP angle {d}    : {bp_angle_sum[d].item() / len(train_loader)}")

    def train_back_weights(self, epoch):
        return

    def compute_target(self, x, y, stepsize):
        y_pred = self.forward(x)

        # initialize
        loss = self.loss_function(y_pred, y)
        for d in range(self.depth):
            if self.layers[d].BNswx.grad is not None:
                self.layers[d].BNswx.grad.zero_()
        loss.backward(retain_graph=True)

        with torch.no_grad():
            D = self.depth - self.direct_depth
            for d in range(D, self.depth):
                self.layers[d].target = self.layers[d].BNswx - stepsize * self.layers[d].BNswx.grad
            for i, d in enumerate(reversed(range(D))):
                self.layers[d].target = self.layers_backward[i].backward(self.layers[D].target)
                self.layers[d].target = self.layers[d].target + self.layers[d].BNswx
                self.layers[d].target = self.layers[d].target - \
                    self.layers_backward[i].backward(self.layers[D].BNswx)
                self.layers[d].target = batch_normalization(self.layers[d].target)

    def update_weights(self, x, lr):
        self.forward(x)
        batch_size = len(x)
        tp_grad = [None] * self.depth
        for d in reversed(range(self.depth)):
            loss = torch.norm(self.layers[d].target - self.layers[d].BNswx)**2
            if self.layers[d].weight.grad is not None:
                self.layers[d].weight.grad.zero_()
            loss.backward(retain_graph=True)
            # alpha = lr / batch_size * ((d + 1) / self.depth)
            alpha = lr / batch_size
            tp_grad[d] = alpha * self.layers[d].weight.grad
            self.layers[d].weight = (self.layers[d].weight - tp_grad[d]).detach().requires_grad_()
        return tp_grad

    def reconstruction_loss(self, x):
        h1 = self.layers[0].forward(x)
        h = h1
        for d in range(1, self.depth - self.direct_depth + 1):
            h = self.layers[d].forward(h)
        for d in range(self.depth - self.direct_depth):
            h = self.layers_backward[d].backward(h)
        return self.MSELoss(h1, h)

    def reconstruction_loss_of_dataset(self, data_loader):
        return torch.tensor(0)
        rec_loss = 0
        for x, y in data_loader:
            x, y = x.to(self.device), y.to(self.device)
            rec_loss = rec_loss + self.reconstruction_loss(x)
        return rec_loss / len(data_loader.dataset)
