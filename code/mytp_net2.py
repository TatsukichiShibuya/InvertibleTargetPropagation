from net import net
from mytp_layer import mytp_layer
from utils import calc_angle, plot_hist_log

import sys
import time
import wandb
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class mytp_net(net):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.direct_depth = kwargs["direct_depth"]
        assert 1 <= self.direct_depth <= self.depth

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

    def train(self, train_loader, valid_loader, epochs, stepsize, lr_ratio, lrb, scaling,
              b_epochs, b_sigma, refinement_iter, refinement_type, b_loss, log):
        if b_loss == "inv":
            b_epochs = 1

        # reconstruction loss
        print(f"rec loss (init): {self.reconstruction_loss_of_dataset(train_loader)}")

        # train forward network
        for e in range(epochs):
            # monitor
            last_weights = [None] * self.depth
            for d in range(self.depth):
                last_weights[d] = self.layers[d].weight
            target_dist = [[] for d in range(self.depth - self.direct_depth)]
            target_dist_plot = [None for i in range(self.depth - self.direct_depth)]
            target_dist_u_plot = [None for i in range(self.depth - self.direct_depth)]
            target_dist_b_plot = [None for i in range(self.depth - self.direct_depth)]
            local_loss_plot = [None for i in range(self.depth)]
            target_angle = [[] for d in range(self.depth - self.direct_depth)]
            monitor_time = 0
            start_time = time.time()

            # train forward
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                # train backward
                for be in range(b_epochs):
                    self.train_backweights(x, lrb, b_sigma, b_loss)

                # compute target
                self.compute_target(x, y, stepsize, refinement_iter, refinement_type)

                ###### monitor start ######
                monitor_start_time = time.time()
                # compute target error
                for d in range(self.depth - self.direct_depth):
                    t = self.layers[d].target
                    for _d in range(d + 1, self.depth - self.direct_depth + 1):
                        t = self.layers[_d].forward(t, update=False)
                    h = self.layers[self.depth - self.direct_depth].linear_activation
                    t_ = self.layers[self.depth - self.direct_depth].target
                    v1, v2, v3 = t - h, t_ - h, t - t_
                    target_angle[d].append(calc_angle(v1, v2).mean())
                    target_dist[d].append(
                        (torch.norm(v3, dim=1) / (torch.norm(v2, dim=1) + 1e-30)).mean())
                    #print(f"targetのずれ {d}: {torch.norm(v3, dim=1).min().item()} {torch.norm(v3, dim=1).max().item()}")

                    local_loss = torch.norm(
                        self.layers[d].linear_activation - self.layers[d].target, dim=1)
                    if target_dist_plot[d] is None:
                        target_dist_plot[d] = torch.norm(v3, dim=1) / torch.norm(v2, dim=1)
                        target_dist_u_plot[d] = torch.norm(v3, dim=1)
                        target_dist_b_plot[d] = torch.norm(v2, dim=1)
                        local_loss_plot[d] = local_loss
                    else:
                        target_dist_plot[d] = torch.cat([target_dist_plot[d],
                                                         torch.norm(v3, dim=1) / torch.norm(v2, dim=1)])
                        target_dist_u_plot[d] = torch.cat([target_dist_u_plot[d],
                                                           torch.norm(v3, dim=1)])
                        target_dist_b_plot[d] = torch.cat([target_dist_b_plot[d],
                                                           torch.norm(v2, dim=1)])
                        local_loss_plot[d] = torch.cat([local_loss_plot[d], local_loss])
                        ratio = local_loss / torch.norm(v1, dim=1)
                        delta = ratio * torch.norm(v3, dim=1)
                        #print("delta      :", d, delta.min(), delta.max(), delta.mean())
                        #print("delta ratio:", d, ratio.min(), ratio.max(), ratio.mean())
                monitor_end_time = time.time()
                monitor_time += monitor_end_time - monitor_start_time
                ###### monitor end ######

                # train forward
                h_2, t_2 = self.layers[2].linear_activation, self.layers[2].target
                move_base = t_2 - h_2

                self.update_weights(x, lr_ratio, scaling=scaling)

                self.forward(x)
                h_2_ = self.layers[2].linear_activation
                move = h_2_ - h_2
                #print("move:", calc_angle(v1, v2).mean(),(torch.norm(move, dim=1) / (torch.norm(move_base, dim=1) + 1e-30)).mean())

            end_time = time.time()
            print(f"epochs {e}: {end_time - start_time - monitor_time:.2f}, {monitor_time:.2f}")

            # predict
            with torch.no_grad():
                train_loss, train_acc = self.test(train_loader)
                valid_loss, valid_acc = self.test(valid_loader)
                rec_loss = self.reconstruction_loss_of_dataset(train_loader)
                if torch.isnan(rec_loss).any():
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
                    for d in range(self.depth):
                        sub = self.MSELoss(self.layers[d].weight, last_weights[d])
                        shape = self.layers[d].weight.shape
                        log_dict[f"weight moving {d}"] = float(sub) / (shape[0] * shape[1])
                    for d in range(self.depth - self.direct_depth):
                        log_dict[f"target error dist {d}"] = torch.mean(
                            torch.tensor(target_dist[d]))
                        log_dict[f"target error angle {d}"] = torch.mean(
                            torch.tensor(target_angle[d]))

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
                    for d in range(self.depth):
                        sub = self.MSELoss(self.layers[d].weight, last_weights[d])
                        shape = self.layers[d].weight.shape
                        print(f"\tweight moving {d}: {float(sub) / (shape[0] * shape[1])}")
                    for d in range(self.depth - self.direct_depth):
                        print(f"\ttarget err dist  {d}: {torch.mean(torch.tensor(target_dist[d]))}")
                        print(
                            f"\ttarget err angle {d}: {torch.mean(torch.tensor(target_angle[d]))}")
                    for d in range(1, self.depth - self.direct_depth + 1):
                        print(f"\tcond {d}: {torch.linalg.cond(self.layers[d].weight)}")

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

                self.layers[d].backweight = (self.layers[d].backweight - lrb *
                                             self.layers[d].backweight.grad).detach().requires_grad_()

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
                for d in reversed(range(self.depth - self.direct_depth)):
                    u = self.layers[d + 1].target
                    for i in range(refinement_iter):
                        gt = self.layers[d + 1].backward(u)
                        fgt = self.layers[d + 1].forward(gt, update=False)
                        u = u + self.layers[d + 1].target - fgt
                    self.layers[d].target = self.layers[d + 1].backward(u)

    def update_weights(self, x, lr_ratio, scaling=False):
        self.forward(x)
        batch_size = len(x)
        for d in reversed(range(self.depth)):
            loss = torch.norm(self.layers[d].target - self.layers[d].linear_activation)**2
            if self.layers[d].weight.grad is not None:
                self.layers[d].weight.grad.zero_()
            loss.backward(retain_graph=True)
            self.layers[d].weight = (self.layers[d].weight - (1 / batch_size) *
                                     self.layers[d].weight.grad).detach().requires_grad_()

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
