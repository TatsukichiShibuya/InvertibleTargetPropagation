from net import net
from dttp_layer import dttp_layer
from utils import calc_angle

import sys
import time
import wandb
import torch
from torch import nn
import numpy as np
from tqdm import tqdm


class dttp_net(net):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.direct_depth = kwargs["direct_depth"]
        assert 1 <= self.direct_depth <= self.depth

        if kwargs["type"][0] == "C":
            self.TRAIN_BACKWARD_TYPE = "DCTP"
        elif kwargs["type"][0] == "T":
            self.TRAIN_BACKWARD_TYPE = "DTTP"

        if kwargs["type"][1] == "C":
            self.TRAIN_FORWARD_TYPE = "DCTP"
        elif kwargs["type"][1] == "T":
            self.TRAIN_FORWARD_TYPE = "DTTP"

        if kwargs["type"][2] == "C":
            self.TARGET_TYPE = "DCTP"
        elif kwargs["type"][2] == "T":
            self.TARGET_TYPE = "DTTP"

    def init_layers(self, in_dim, hid_dim, out_dim, activation_function):
        layers = [None] * self.depth

        # first layer
        layers[0] = dttp_layer(in_dim, hid_dim, activation_function, self.device)
        # hidden layers
        for i in range(1, self.depth - 1):
            layers[i] = dttp_layer(hid_dim, hid_dim, activation_function, self.device)
        # last layer
        #layers[-1] = dttp_layer(hid_dim, out_dim, "linear", self.device)
        layers[-1] = dttp_layer(hid_dim, out_dim, activation_function, self.device)

        return layers

    def train(self, train_loader, valid_loader, epochs, stepsize, lr_ratio, lrf, lrb, scaling,
              b_epochs, b_sigma, refinement_iter, log):
        # train backward network
        for e in range(10):
            # train backward
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                for be in range(b_epochs):
                    self.train_backweights(x, lrb, b_sigma)

            # reconstruction loss
            rec_loss = self.reconstruction_loss_of_dataset(train_loader)
            if torch.isnan(rec_loss).any():
                print("ERROR: rec loss diverged")
                sys.exit(1)
            print(f"epochs {e}: {rec_loss}")

        # train forward network
        for e in range(epochs):
            torch.cuda.empty_cache()
            # monitor
            refinement_converge = [[] for d in range(self.depth - self.direct_depth)]
            target_ratio_list = [torch.tensor([], device=self.device)
                                 for d in range(self.depth - self.direct_depth)]
            target_angle_list = [torch.tensor([], device=self.device)
                                 for d in range(self.depth - self.direct_depth)]

            move_ratio_DCTP_list = [torch.tensor([], device=self.device)
                                    for d in range(self.depth)]
            move_angle_DCTP_list = [torch.tensor([], device=self.device)
                                    for d in range(self.depth)]
            move_ratio_DTTP_list = [torch.tensor([], device=self.device)
                                    for d in range(self.depth)]
            move_angle_DTTP_list = [torch.tensor([], device=self.device)
                                    for d in range(self.depth)]

            monitor_time = 0
            start_time = time.time()

            # train backward
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                for be in range(b_epochs):
                    self.train_backweights(x, lrb, b_sigma)
            rec_loss = self.reconstruction_loss_of_dataset(train_loader)
            if torch.isnan(rec_loss).any():
                print("ERROR: rec loss diverged")
                sys.exit(1)
            print(f"before epochs {e}:\n\trec loss       : {rec_loss}")

            # train forward
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                """
                # train backward
                for be in range(b_epochs):
                    self.train_backweights(x, lrb, b_sigma)
                """

                # compute target
                self.compute_target(x, y, stepsize, refinement_iter)

                ###### monitor start ######
                monitor_start_time = time.time()
                """
                ret = self.check_refinement()
                for d in range(self.depth - self.direct_depth):
                    refinement_converge[d].append(ret[d])
                """
                D = self.depth - self.direct_depth
                for d1 in range(D):
                    t = self.layers[d1].target
                    for d2 in range(d1 + 1, D + 1):
                        t = self.layers[d2].forward(t, update=False)
                    v1 = self.layers[D].linear_activation - t
                    v2 = self.layers[D].linear_activation - self.layers[D].target
                    nonzero = (torch.norm(v2, dim=1) > 1e-6)
                    target_ratio = torch.norm(v1[nonzero], dim=1) / torch.norm(v2[nonzero], dim=1)
                    target_ratio_list[d1] = torch.cat([target_ratio_list[d1], target_ratio])
                    target_angle = calc_angle(v1[nonzero], v2[nonzero])
                    target_angle_list[d1] = torch.cat([target_angle_list[d1], target_angle])
                monitor_end_time = time.time()
                monitor_time = monitor_time + monitor_end_time - monitor_start_time
                ###### monitor end ######

                # train forward
                move_ratio_DCTP, move_ratio_DTTP, move_angle_DCTP, move_angle_DTTP = self.update_weights(
                    x, lrf, lr_ratio, scaling)

                ###### monitor start ######
                monitor_start_time = time.time()
                for d in range(self.depth):
                    move_ratio_DCTP_list[d] = torch.cat([move_ratio_DCTP_list[d],
                                                         move_ratio_DCTP[d]])
                    move_ratio_DTTP_list[d] = torch.cat([move_ratio_DTTP_list[d],
                                                         move_ratio_DTTP[d]])
                    move_angle_DCTP_list[d] = torch.cat([move_angle_DCTP_list[d],
                                                         move_angle_DCTP[d]])
                    move_angle_DTTP_list[d] = torch.cat([move_angle_DTTP_list[d],
                                                         move_angle_DTTP[d]])
                monitor_end_time = time.time()
                monitor_time = monitor_time + monitor_end_time - monitor_start_time
                ###### monitor end ######

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
                    for d in range(self.depth - self.direct_depth):
                        """
                        x = torch.tensor(refinement_converge[d])
                        log_dict[f"convergence {d}"] = (torch.sum(x) / len(x)).item()
                        """
                        log_dict[f"target ratio {d}"] = torch.mean(target_ratio_list[d]).item()
                        log_dict[f"target angle {d}"] = torch.mean(target_angle_list[d]).item()
                    for d in range(self.depth):
                        log_dict[f"move ratio DCTP {d}"] = torch.mean(
                            move_ratio_DCTP_list[d]).item()
                        log_dict[f"move angle DCTP {d}"] = torch.mean(
                            move_angle_DCTP_list[d]).item()
                        log_dict[f"move ratio DTTP {d}"] = torch.mean(
                            move_ratio_DTTP_list[d]).item()
                        log_dict[f"move angle DTTP {d}"] = torch.mean(
                            move_angle_DTTP_list[d]).item()

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
                    for d in range(self.depth - self.direct_depth):
                        """
                        x = torch.tensor(refinement_converge[d])
                        print(f"\tconvergence {d}: {(torch.sum(x) / len(x)).item()}")
                        """
                        print(f"\ttarget ratio {d}: {torch.mean(target_ratio_list[d]).item()}")
                        print(f"\ttarget angle {d}: {torch.mean(target_angle_list[d]).item()}")

                    for d in range(self.depth):
                        print(
                            f"\tmove ratio DCTP {d}: {torch.mean(move_ratio_DCTP_list[d]).item()}")
                        print(
                            f"\tmove angle DCTP {d}: {torch.mean(move_angle_DCTP_list[d]).item()}")
                        print(
                            f"\tmove ratio DTTP {d}: {torch.mean(move_ratio_DTTP_list[d]).item()}")
                        print(
                            f"\tmove angle DTTP {d}: {torch.mean(move_angle_DTTP_list[d]).item()}")

    def check_refinement(self):
        ret = []
        for d in range(self.depth - self.direct_depth):
            y = self.layers[d + 1].linear_activation
            gy = self.layers[d + 1].backward(y)
            loss_before = torch.norm(gy - self.layers[d].linear_activation, dim=1)

            x = self.layers[d + 1].backward(y)
            for i in range(10):
                fx = self.layers[d + 1].forward(x, update=False)
                gfx = self.layers[d + 1].backward(fx)
                x = x + gy - gfx
            loss_after = torch.norm(x - self.layers[d].linear_activation, dim=1)

            flag1 = (loss_after < loss_before)
            flag2 = (loss_after < 1e-3)
            ret.append(torch.logical_or(flag1, flag2).all().item())
        return ret

    def train_backweights(self, x, lrb, b_sigma):
        if self.TRAIN_BACKWARD_TYPE == "DCTP":
            self.train_backweights_DCTP(x, lrb, b_sigma)
        elif self.TRAIN_BACKWARD_TYPE == "DTTP":
            self.train_backweights_DTTP(x, lrb, b_sigma)

    def train_backweights_DCTP(self, x, lrb, b_sigma):
        self.forward(x)
        batch_size = len(x)
        for d in reversed(range(1, self.depth - self.direct_depth + 1)):
            q = self.layers[d - 1].linear_activation.detach().clone()
            q = q + torch.normal(0, b_sigma, size=q.shape, device=self.device)
            h = self.layers[d].backward(self.layers[d].forward(q, update=False))
            loss = self.MSELoss(h, q)
            if self.layers[d].backweight.grad is not None:
                self.layers[d].backweight.grad.zero_()
            loss.backward(retain_graph=True)
            self.layers[d].backweight = (self.layers[d].backweight -
                                         (lrb / batch_size) * self.layers[d].backweight.grad).detach().requires_grad_()

    def train_backweights_DTTP(self, x, lrb, b_sigma):
        self.forward(x)
        for d in reversed(range(1, self.depth - self.direct_depth + 1)):
            q = self.layers[d - 1].linear_activation.detach().clone()
            q = q + torch.normal(0, b_sigma, size=q.shape, device=self.device)
            q_next = self.layers[d].forward(q, update=False)
            s = self.layers[d].activation_function(q_next)
            n = s / (s**2).sum(axis=1).reshape(-1, 1)
            grad = (q - self.layers[d].backward(q_next)).T @ (n * lrb)
            if not (torch.isnan(grad).any() or torch.isinf(grad).any()):
                self.layers[d].backweight = (
                    self.layers[d].backweight + grad).detach().requires_grad_()

    def compute_target(self, x, y, stepsize, refinement_iter):
        if self.TARGET_TYPE == "DCTP":
            self.compute_target_DCTP(x, y, stepsize, refinement_iter)
        elif self.TARGET_TYPE == "DTTP":
            self.compute_target_DTTP(x, y, stepsize, refinement_iter)

    def compute_target_DCTP(self, x, y, stepsize, refinement_iter):
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
                self.layers[d].target = self.layers[d].target + self.layers[d].linear_activation
                self.layers[d].target = self.layers[d].target - \
                    self.layers[d + 1].backward(self.layers[d + 1].linear_activation)

    def compute_target_DTTP(self, x, y, stepsize, refinement_iter):
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

            for i in range(refinement_iter):
                for d in reversed(range(self.depth - self.direct_depth)):
                    gt = self.layers[d + 1].backward(self.layers[d + 1].target)
                    ft = self.layers[d + 1].forward(self.layers[d].target, update=False)
                    gft = self.layers[d + 1].backward(ft)
                    self.layers[d].target = self.layers[d].target + gt - gft

    def update_weights(self, x, lrf, lr_ratio, scaling=False):
        move_target = [None for d in range(self.depth)]
        move_DCTP = [None for d in range(self.depth)]
        move_DTTP = [None for d in range(self.depth)]
        for d in range(self.depth):
            move_target[d] = self.layers[d].target - self.layers[d].linear_activation

        weights = [None for d in range(self.depth)]
        for d in range(self.depth):
            weights[d] = self.layers[d].weight

        if self.TRAIN_FORWARD_TYPE == "DCTP":
            self.update_weights_DTTP(x, lrf, lr_ratio, scaling=False)
            for d in range(self.depth):
                h_previous = self.layers[d - 1].linear_activation if d > 0 else x
                h_after = self.layers[d].forward(h_previous, update=False)
                move_DTTP[d] = h_after - self.layers[d].linear_activation

            for d in range(self.depth):
                self.layers[d].weight = weights[d].detach().requires_grad_()

            self.update_weights_DCTP(x, lrf, lr_ratio, scaling=False)
            for d in range(self.depth):
                h_previous = self.layers[d - 1].linear_activation if d > 0 else x
                h_after = self.layers[d].forward(h_previous, update=False)
                move_DCTP[d] = h_after - self.layers[d].linear_activation
        elif self.TRAIN_FORWARD_TYPE == "DTTP":
            self.update_weights_DCTP(x, lrf, lr_ratio, scaling=False)
            for d in range(self.depth):
                h_previous = self.layers[d - 1].linear_activation if d > 0 else x
                h_after = self.layers[d].forward(h_previous, update=False)
                move_DCTP[d] = h_after - self.layers[d].linear_activation

            for d in range(self.depth):
                self.layers[d].weight = weights[d].detach().requires_grad_()

            self.update_weights_DTTP(x, lrf, lr_ratio, scaling=False)
            for d in range(self.depth):
                h_previous = self.layers[d - 1].linear_activation if d > 0 else x
                h_after = self.layers[d].forward(h_previous, update=False)
                move_DTTP[d] = h_after - self.layers[d].linear_activation

        move_ratio_DCTP = [None for d in range(self.depth)]
        move_ratio_DTTP = [None for d in range(self.depth)]
        move_angle_DCTP = [None for d in range(self.depth)]
        move_angle_DTTP = [None for d in range(self.depth)]
        for d in range(self.depth):
            move_target_norm = torch.norm(move_target[d], dim=1)
            nonzero = (move_target_norm > 1e-6)
            move_DCTP_norm = torch.norm(move_DCTP[d], dim=1)
            move_DTTP_norm = torch.norm(move_DTTP[d], dim=1)
            move_ratio_DCTP[d] = move_DCTP_norm[nonzero] / move_target_norm[nonzero]
            move_ratio_DTTP[d] = move_DTTP_norm[nonzero] / move_target_norm[nonzero]
            move_angle_DCTP[d] = calc_angle(move_DCTP[d][nonzero], move_target[d][nonzero])
            move_angle_DTTP[d] = calc_angle(move_DTTP[d][nonzero], move_target[d][nonzero])

        return move_ratio_DCTP, move_ratio_DTTP, move_angle_DCTP, move_angle_DTTP

    def update_weights_DCTP(self, x, lrf, lr_ratio, scaling=False):
        self.forward(x)
        batch_size = len(x)
        for d in reversed(range(self.depth)):
            loss = torch.norm(self.layers[d].target - self.layers[d].linear_activation)**2
            if self.layers[d].weight.grad is not None:
                self.layers[d].weight.grad.zero_()
            loss.backward(retain_graph=True)
            self.layers[d].weight = (self.layers[d].weight -
                                     (lrf / batch_size) * self.layers[d].weight.grad).detach().requires_grad_()

    def update_weights_DTTP(self, x, lrf, lr_ratio, scaling=False):
        self.forward(x)
        D = self.depth - self.direct_depth
        global_loss = ((self.layers[D].target - self.layers[D].linear_activation)**2).sum(axis=1)
        grad_base = 0
        batch_size = len(x)
        for d in reversed(range(self.depth)):
            # compute grad
            local_loss = ((self.layers[d].target - self.layers[d].linear_activation)**2).sum(axis=1)
            if self.TARGET_TYPE == "DTTP" and d < D:
                lr = (global_loss / (local_loss + 1e-30)).reshape(-1, 1)
            else:
                lr = torch.tensor(1.)
            n = self.layers[d].activation / \
                (self.layers[d].activation**2).sum(axis=1).reshape(-1, 1)
            grad = (self.layers[d].target -
                    self.layers[d].linear_activation).T @ (n * lr * lrf)

            # update weight
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
            rec_loss = rec_loss + self.reconstruction_loss(x)
        return rec_loss / len(data_loader.dataset)
