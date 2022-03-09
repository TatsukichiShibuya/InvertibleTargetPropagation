from net import net
from invtp_layer import invtp_layer
from utils import calc_angle, batch_normalization, batch_normalization_inverse, get_seed

import sys
import time
import wandb
import torch
from torch import nn
import numpy as np


class invtp_net(net):
    def __init__(self, **kwargs):
        self.seed = kwargs["seed"]
        super().__init__(**kwargs)
        self.direct_depth = kwargs["direct_depth"]
        assert 1 <= self.direct_depth <= self.depth

    def init_layers(self, in_dim, hid_dim, out_dim, activation_function):
        layers = [None] * self.depth

        # first layer
        layers[0] = invtp_layer(in_dim, hid_dim, activation_function, self.device,
                                0 + self.seed * 11)
        # hidden layers
        for d in range(1, self.depth - 1):
            layers[d] = invtp_layer(hid_dim, hid_dim, activation_function, self.device,
                                    d + self.seed * 11)
        # last layer
        layers[-1] = invtp_layer(hid_dim, out_dim, activation_function, self.device,
                                 self.depth - 1 + self.seed)

        return layers

    def train(self, train_loader, valid_loader, epochs, stepsize, lr, log):
        # reconstruction loss
        rec_loss = self.reconstruction_loss_of_dataset(train_loader)
        print(f"Initial Rec Loss: {rec_loss}")

        layerwise_loss_sum = [0] * (self.depth - self.direct_depth)
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.forward(x)
            with torch.no_grad():
                D = self.depth - self.direct_depth
                h = self.layers[0].BNswx
                for d in range(1, D + 1):
                    h_upper = self.layers[d].forward(h, update=False)
                    h_rec = self.layers[d].backward(h_upper)
                    rec_loss = torch.norm(h_rec - h, dim=1)
                    layerwise_loss_sum[d - 1] = layerwise_loss_sum[d - 1] + rec_loss.sum()
                    h = h_upper
        datasize = len(train_loader.dataset)
        for d in range(self.depth - self.direct_depth):
            print(f"\tRec Loss {d+1}: {layerwise_loss_sum[d].item() / datasize}")

        # train forward network
        for e in range(epochs):
            torch.cuda.empty_cache()
            target_ratio_sum = [0] * (self.depth - self.direct_depth)
            target_angle_sum = [0] * (self.depth - self.direct_depth)

            tp_angle_sum = [0] * (self.depth - self.direct_depth)
            tp_ratio_sum = [0] * (self.depth - self.direct_depth)
            dtp_angle_sum = [0] * (self.depth - self.direct_depth)
            dtp_ratio_sum = [0] * (self.depth - self.direct_depth)
            rtp_angle_sum = [0] * (self.depth - self.direct_depth)
            rtp_ratio_sum = [0] * (self.depth - self.direct_depth)

            layerwise_loss_sum = [0] * (self.depth - self.direct_depth)

            bp_angle_sum = [0] * self.depth

            monitor_time = 0

            # train backward
            self.train_back_weights(e)

            rec_loss = self.reconstruction_loss_of_dataset(train_loader)
            print(f"Before Epoch {e}:\n\tRec Loss       : {rec_loss}")

            start_time = time.time()

            # train forward
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                # compute target
                self.compute_target(x, y, stepsize)

                ###### monitor start ######
                monitor_start_time = time.time()

                # compute target-angle
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

                # compute Layerwize Reconstruction Loss
                with torch.no_grad():
                    D = self.depth - self.direct_depth
                    h = self.layers[0].BNswx
                    for d in range(1, D + 1):
                        h_upper = self.layers[d].forward(h, update=False)
                        h_rec = self.layers[d].backward(h_upper)
                        rec_loss = torch.norm(h_rec - h, dim=1)
                        layerwise_loss_sum[d - 1] = layerwise_loss_sum[d - 1] + rec_loss.sum()
                        h = h_upper

                # compute update-angle
                with torch.no_grad():
                    D = self.depth - self.direct_depth
                    for d in range(D):
                        t_upper = self.layers[d + 1].target
                        h_upper = self.layers[d + 1].BNswx
                        h = self.layers[d].BNswx

                        v1 = self.layers[d + 1].backward(h_upper) - h
                        v2 = self.layers[d + 1].backward(t_upper) - h
                        v3 = v2 - v1
                        v4 = self.layers[d].target - h

                        tp_angle_sum[d] = tp_angle_sum[d] + calc_angle(v1, v2).sum()
                        dtp_angle_sum[d] = dtp_angle_sum[d] + calc_angle(v1, v3).sum()
                        rtp_angle_sum[d] = rtp_angle_sum[d] + calc_angle(v1, v4).sum()

                        nonzero = torch.norm(v1, dim=1) > 1e-6
                        v1_norm = torch.norm(v1[nonzero], dim=1)
                        v2_norm = torch.norm(v2[nonzero], dim=1)
                        v3_norm = torch.norm(v3[nonzero], dim=1)
                        v4_norm = torch.norm(v4[nonzero], dim=1)
                        tp_ratio_sum[d] = tp_ratio_sum[d] + (v2_norm / v1_norm).sum()
                        dtp_ratio_sum[d] = dtp_ratio_sum[d] + (v3_norm / v1_norm).sum()
                        rtp_ratio_sum[d] = rtp_ratio_sum[d] + (v4_norm / v1_norm).sum()

                # compute BP-angle
                y_pred = self.forward(x)
                loss = self.loss_function(y_pred, y)
                for d in range(self.depth):
                    self.weights_zero_grad(d)
                loss.backward()
                bp_grad = [None] * self.depth
                for d in range(self.depth):
                    bp_grad[d] = self.layers[d].weight.grad.clone()

                monitor_end_time = time.time()
                monitor_time = monitor_time + monitor_end_time - monitor_start_time
                ###### monitor end ######

                # train forward
                tp_grad = self.update_weights(x, lr)

                ###### monitor start ######
                monitor_start_time = time.time()
                with torch.no_grad():
                    for d in range(self.depth):
                        bp_angle_sum[d] += calc_angle(tp_grad[d].reshape((1, -1)),
                                                      bp_grad[d].reshape((1, -1))).sum()

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

                        log_dict[f"TP ratio {d}"] = tp_ratio_sum[d].item() / datasize
                        log_dict[f"TP angle {d}"] = tp_angle_sum[d].item() / datasize
                        log_dict[f"DTP ratio {d}"] = dtp_ratio_sum[d].item() / datasize
                        log_dict[f"DTP angle {d}"] = dtp_angle_sum[d].item() / datasize
                        log_dict[f"RTP ratio {d}"] = rtp_ratio_sum[d].item() / datasize
                        log_dict[f"RTP angle {d}"] = rtp_angle_sum[d].item() / datasize

                        log_dict[f"Rec Loss {d+1}"] = layerwise_loss_sum[d].item() / datasize

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
                        print(f"\tTP ratio {d}: {tp_ratio_sum[d].item() / datasize}")
                        print(f"\tTP angle {d}: {tp_angle_sum[d].item() / datasize}")
                        print(f"\tDTP ratio {d}: {dtp_ratio_sum[d].item() / datasize}")
                        print(f"\tDTP angle {d}: {dtp_angle_sum[d].item() / datasize}")
                        print(f"\tRTP ratio {d}: {rtp_ratio_sum[d].item() / datasize}")
                        print(f"\tRTP angle {d}: {rtp_angle_sum[d].item() / datasize}")

                    for d in range(self.depth - self.direct_depth):
                        print(f"\tRec Loss {d+1}: {layerwise_loss_sum[d].item() / datasize}")

                    for d in range(self.depth):
                        print(f"\tBP angle {d}    : {bp_angle_sum[d].item() / len(train_loader)}")

    def train_back_weights(self, epoch):
        return

    def compute_target(self, x, y, stepsize):
        y_pred = self.forward(x)

        # initialize
        loss = self.loss_function(y_pred, y)
        for d in range(self.depth):
            self.activations_zero_grad(d)
        loss.backward(retain_graph=True)

        with torch.no_grad():
            for d in range(self.depth - self.direct_depth, self.depth):
                self.layers[d].target = self.layers[d].BNswx - stepsize * self.layers[d].BNswx.grad

            for d in reversed(range(self.depth - self.direct_depth)):
                self.layers[d].target = self.layers[d + 1].backward(self.layers[d + 1].target)
                self.layers[d].target = self.layers[d].target + self.layers[d].BNswx
                self.layers[d].target = self.layers[d].target - \
                    self.layers[d + 1].backward(self.layers[d + 1].BNswx)
                self.layers[d].target = batch_normalization(self.layers[d].target)

    def update_weights(self, x, lr):
        self.forward(x)
        batch_size = len(x)
        tp_grad = [None] * self.depth
        for d in reversed(range(self.depth)):
            t, h = self.layers[d].target, self.layers[d].BNswx
            base_loss, rec_loss = torch.norm(t - h)**2, torch.tensor(0)
            """if d > 0:
                h_rec = self.layers[d].backward(h)
                h_prev = self.layers[d - 1].BNswx
                rec_loss = torch.norm(h_prev - h_rec)**2
            if d < self.depth - 1:
                h_upper = self.layers[d + 1].BNswx
                h_rec = self.layers[d + 1].backward(h_upper)
                rec_loss = torch.norm(h - h_rec)**2
            ratio = base_loss.item() / (rec_loss.item() + 1e-12)
            loss = base_loss + 10 * ratio * rec_loss
            """
            loss = base_loss
            self.weights_zero_grad(d)
            loss.backward(retain_graph=True)
            alpha = lr / batch_size
            tp_grad[d] = alpha * self.layers[d].weight.grad
            self.layers[d].weight = (self.layers[d].weight - tp_grad[d]).detach().requires_grad_()
        return tp_grad

    def reconstruction_loss(self, x):
        h1 = self.layers[0].forward(x)
        h = h1
        for d in range(1, self.depth - self.direct_depth + 1):
            h = self.layers[d].forward(h)
        for d in reversed(range(1, self.depth - self.direct_depth + 1)):
            h = self.layers[d].backward(h)
        return self.MSELoss(h1, h)

    def reconstruction_loss_of_dataset(self, data_loader):
        rec_loss = 0
        for x, y in data_loader:
            x, y = x.to(self.device), y.to(self.device)
            rec_loss = rec_loss + self.reconstruction_loss(x)
        if torch.isnan(rec_loss).any():
            print("ERROR: rec loss diverged")
            sys.exit(1)
        return rec_loss / len(data_loader.dataset)

    def weights_zero_grad(self, d):
        if self.layers[d].weight.grad is not None:
            self.layers[d].weight.grad.zero_()

    def activations_zero_grad(self, d):
        if self.layers[d].BNswx.grad is not None:
            self.layers[d].BNswx.grad.zero_()
