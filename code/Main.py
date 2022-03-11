from utils import plot_regression, worker_init_fn, fix_seed, combined_loss
from dataset import make_regression_dataset, make_MNIST, make_CIFAR10, make_fashionMNIST, make_CIFAR100

from bp_net import bp_net
from dttp_net import dttp_net
from mytp_net2 import mytp_net
from invtp_net import invtp_net
from ditp_net import ditp_net

import os
import sys
import wandb
import torch
import argparse
import numpy as np
from torch import nn

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--problem",    type=str, default="classification",
                        choices=['regression', 'MNIST', 'CIFAR10', "fashionMNIST", "CIFAR100"])
    parser.add_argument("--label_augmentation", action="store_true")
    parser.add_argument("--datasize",   type=int, default=70000)

    # model architecture
    parser.add_argument("--depth",      type=int, default=6)
    parser.add_argument("--in_dim",     type=int, default=784)
    parser.add_argument("--hid_dim",    type=int, default=256)
    parser.add_argument("--out_dim",    type=int, default=10)
    parser.add_argument("--activation_function", type=str, default="tanh",
                        choices=['leakyrelu', 'sigmoid', 'relu', 'tanh', 'linear'])
    parser.add_argument("--back_activation_function", type=str, default="tanh",
                        choices=['leakyrelu', 'sigmoid', 'relu', 'tanh', 'linear'])
    # learning algorithm
    parser.add_argument("--algorithm",  type=str, default="BP",
                        choices=['BP', 'DTTP', 'MyTP', 'InvTP', "DITP", "FA"])
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--batch_size",  type=int, default=128)
    parser.add_argument("--seed",       type=int, default=1)

    parser.add_argument("--init_dist",  type=str, default="uniform",
                        choices=["uniform", "gaussian", "eye"])
    parser.add_argument("--init_range", type=float, default=1e-4)  # uniform
    parser.add_argument("--init_mean",  type=float, default=0)     # gaussian
    parser.add_argument("--init_std",   type=float, default=1e-4)  # gaussian

    parser

    # parameters used in BP
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-6)

    # parameters used in DTTP and MyTP
    parser.add_argument("--stepsize",   type=float, default=1e-2)
    parser.add_argument("--lr_ratio",   type=float, default=1)
    parser.add_argument("--weight_scaling", action="store_true")
    parser.add_argument("--b_epochs",   type=int, default=5)
    parser.add_argument("--b_sigma",    type=float, default=0.08)
    parser.add_argument("--b_loss",     type=str, default="gf",
                        choices=['gf', 'fg', 'eye', 'inv'])
    parser.add_argument("--direct_depth", type=int, default=2)
    parser.add_argument("--refinement_iter", type=int, default=5)
    parser.add_argument("--refinement_type", type=str, default="gf",
                        choices=['gf', 'fg'])
    parser.add_argument("--learning_rate_for_backward", "-lrb", type=float, default=1e-2)

    # other
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--agent", action="store_true")
    parser.add_argument("--type",  type=str, default="CCC", choices=['CCC', 'CCT', 'CTC', 'CTT',
                                                                     'TCC', 'TCT', 'TTC', 'TTT',
                                                                     'ICC', 'ICT', 'ITC', 'ITT',
                                                                     'T', 'C'])

    args = parser.parse_args()
    return args


def main(**kwargs):
    # initialize
    fix_seed(kwargs["seed"])

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        os.environ['OMP_NUM_THREADS'] = '1'
    print(f"DEVICE: {device}")

    if kwargs["log"]:
        if kwargs["agent"]:
            config = {"problem": kwargs["problem"] + "-" + str(kwargs["datasize"]),
                      "algorithm": kwargs["algorithm"],
                      "epochs": kwargs["epochs"],
                      "seed": kwargs["seed"],
                      "depth": kwargs["depth"],
                      "in_dim": kwargs["in_dim"],
                      "out_dim": kwargs["out_dim"],
                      "activation function": kwargs["activation_function"]}
            if kwargs["algorithm"] in ["DTTP", "MyTP"]:
                config["direct depth"] = kwargs["direct_depth"]
                config["stepsize"] = kwargs["stepsize"]
                config["lr ratio"] = kwargs["lr_ratio"]
                config["learning rate (backward)"] = kwargs["learning_rate_for_backward"]
                config["epochs (backward)"] = kwargs["b_epochs"]
                config["sigma (backward)"] = kwargs["b_sigma"]
                config["refinement iteration"] = kwargs["refinement_iter"]
                if kwargs["algorithm"] == "MyTP":
                    config["refinement type"] = kwargs["refinement_type"]
                    config["loss (backward)"] = kwargs["b_loss"]
            wandb.init(config=config)
        else:
            wandb.init(project="InvertibleTargetPropagation", entity="tatsukichishibuya")
            config = {"problem": kwargs["problem"] + "-" + str(kwargs["datasize"]),
                      "algorithm": kwargs["algorithm"],
                      "epochs": kwargs["epochs"],
                      "batch_size": kwargs["batch_size"],
                      "seed": kwargs["seed"],
                      "depth": kwargs["depth"],
                      "in_dim": kwargs["in_dim"],
                      "hid_dim": kwargs["hid_dim"],
                      "out_dim": kwargs["out_dim"],
                      "activation function": kwargs["activation_function"]}
            if kwargs["algorithm"] == "BP":
                config["learning rate"] = kwargs["learning_rate"]
            elif kwargs["algorithm"] in ["DTTP", "MyTP", "InvTP", "DITP"]:
                config["direct depth"] = kwargs["direct_depth"]
                config["stepsize"] = kwargs["stepsize"]
                config["lr ratio"] = kwargs["lr_ratio"]
                config["learning rate (backward)"] = kwargs["learning_rate_for_backward"]
                config["epochs (backward)"] = kwargs["b_epochs"]
                config["sigma (backward)"] = kwargs["b_sigma"]
                config["refinement iteration"] = kwargs["refinement_iter"]
                config["learning rate"] = kwargs["learning_rate"]
                if kwargs["algorithm"] == "MyTP":
                    config["refinement type"] = kwargs["refinement_type"]
                    config["loss (backward)"] = kwargs["b_loss"]
            elif kwargs["algorithm"] == "InvTP":
                config["direct depth"] = kwargs["direct_depth"]
                config["stepsize"] = kwargs["stepsize"]
                config["refinement iteration"] = kwargs["refinement_iter"]
                config["learning rate"] = kwargs["learning_rate"]
            wandb.init(config=config)

    # make dataset
    if kwargs["problem"] == "regression":
        trainset, validset, testset = make_regression_dataset(kwargs["datasize"], kwargs["in_dim"])
        loss_function = nn.MSELoss(reduction="sum")
    else:
        if kwargs["problem"] == "MNIST":
            num_classes = 10
            trainset, validset, testset = make_MNIST(kwargs["datasize"],
                                                     kwargs["label_augmentation"],
                                                     kwargs["out_dim"])
        elif kwargs["problem"] == "CIFAR10":
            num_classes = 10
            trainset, validset, testset = make_CIFAR10(kwargs["datasize"],
                                                       kwargs["label_augmentation"],
                                                       kwargs["out_dim"])
        elif kwargs["problem"] == "CIFAR100":
            num_classes = 100
            trainset, validset, testset = make_CIFAR100(kwargs["datasize"],
                                                        kwargs["label_augmentation"],
                                                        kwargs["out_dim"])
        elif kwargs["problem"] == "fashionMNIST":
            num_classes = 10
            trainset, validset, testset = make_fashionMNIST(kwargs["datasize"],
                                                            kwargs["label_augmentation"],
                                                            kwargs["out_dim"])
        if kwargs["label_augmentation"]:
            loss_function = (lambda pred, label: combined_loss(pred, label, device, num_classes))
        else:
            loss_function = nn.CrossEntropyLoss(reduction="sum")

    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=kwargs["batch_size"],
                                               shuffle=True,
                                               num_workers=2,
                                               pin_memory=True,
                                               worker_init_fn=worker_init_fn)
    valid_loader = torch.utils.data.DataLoader(validset,
                                               batch_size=kwargs["batch_size"],
                                               shuffle=False,
                                               num_workers=2,
                                               pin_memory=True,
                                               worker_init_fn=worker_init_fn)
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=kwargs["batch_size"],
                                              shuffle=False,
                                              num_workers=2,
                                              pin_memory=True,
                                              worker_init_fn=worker_init_fn)

    # initialize model
    if kwargs["algorithm"] == "BP":
        model = bp_net(device=device,
                       depth=kwargs["depth"],
                       in_dim=kwargs["in_dim"],
                       out_dim=kwargs["out_dim"],
                       hid_dim=kwargs["hid_dim"],
                       activation_function=kwargs["activation_function"],
                       loss_function=loss_function)
    elif kwargs["algorithm"] == "FA":
        model = bp_net(device=device,
                       depth=kwargs["depth"],
                       in_dim=kwargs["in_dim"],
                       out_dim=kwargs["out_dim"],
                       hid_dim=kwargs["hid_dim"],
                       activation_function=kwargs["activation_function"],
                       loss_function=loss_function)
    elif kwargs["algorithm"] == "DTTP":
        model = dttp_net(device=device,
                         depth=kwargs["depth"],
                         in_dim=kwargs["in_dim"],
                         out_dim=kwargs["out_dim"],
                         hid_dim=kwargs["hid_dim"],
                         direct_depth=kwargs["direct_depth"],
                         activation_function=kwargs["activation_function"],
                         loss_function=loss_function,
                         type=kwargs["type"])
    elif kwargs["algorithm"] == "MyTP":
        model = mytp_net(device=device,
                         depth=kwargs["depth"],
                         in_dim=kwargs["in_dim"],
                         out_dim=kwargs["out_dim"],
                         hid_dim=kwargs["hid_dim"],
                         direct_depth=kwargs["direct_depth"],
                         activation_function=kwargs["activation_function"],
                         loss_function=loss_function,
                         type=kwargs["type"])
    elif kwargs["algorithm"] == "InvTP":
        model = invtp_net(device=device,
                          depth=kwargs["depth"],
                          in_dim=kwargs["in_dim"],
                          out_dim=kwargs["out_dim"],
                          hid_dim=kwargs["hid_dim"],
                          direct_depth=kwargs["direct_depth"],
                          activation_function=kwargs["activation_function"],
                          back_activation_function=kwargs["back_activation_function"],
                          loss_function=loss_function,
                          init_params=make_init_params(**kwargs),
                          seed=kwargs["seed"])
    elif kwargs["algorithm"] == "DITP":
        model = ditp_net(device=device,
                         depth=kwargs["depth"],
                         in_dim=kwargs["in_dim"],
                         out_dim=kwargs["out_dim"],
                         hid_dim=kwargs["hid_dim"],
                         direct_depth=kwargs["direct_depth"],
                         activation_function=kwargs["activation_function"],
                         loss_function=loss_function)

    # train
    if kwargs["algorithm"] == "BP":
        model.train(train_loader, valid_loader, kwargs["epochs"], kwargs["learning_rate"],
                    log=kwargs["log"])
    elif kwargs["algorithm"] == "FA":
        model.train(train_loader, valid_loader, kwargs["epochs"], kwargs["learning_rate"],
                    log=kwargs["log"])
    elif kwargs["algorithm"] == "DTTP":
        model.train(train_loader, valid_loader, kwargs["epochs"], kwargs["stepsize"],
                    kwargs["lr_ratio"], kwargs["learning_rate"], kwargs["learning_rate_for_backward"],
                    kwargs["weight_scaling"], kwargs["b_epochs"], kwargs["b_sigma"],
                    kwargs["refinement_iter"], kwargs["log"])
    elif kwargs["algorithm"] == "MyTP":
        model.train(train_loader, valid_loader, kwargs["epochs"], kwargs["stepsize"],
                    kwargs["lr_ratio"], kwargs["learning_rate"], kwargs["learning_rate_for_backward"],
                    kwargs["weight_scaling"], kwargs["b_epochs"], kwargs["b_sigma"],
                    kwargs["refinement_iter"], kwargs["log"])
    elif kwargs["algorithm"] == "InvTP":
        model.train(train_loader, valid_loader, kwargs["epochs"], kwargs["stepsize"],
                    kwargs["learning_rate"], kwargs["log"])
    elif kwargs["algorithm"] == "DITP":
        model.train(train_loader, valid_loader, kwargs["epochs"], kwargs["stepsize"],
                    kwargs["learning_rate"], kwargs["log"])

    # test
    print(f"\ttest  : {model.test(test_loader)}")

    # plot
    if kwargs["plot"] and (["problem"] == "regression" and kwargs["in_dim"] == 2):
        plot_regression(model, kwargs["algorithm"])


def make_init_params(**kwargs):
    init_params = {}
    init_params["dist"] = kwargs["init_dist"]
    if init_params["dist"] == "uniform":
        init_params["range"] = kwargs["init_range"]
    elif init_params["dist"] == "gaussian":
        init_params["mean"] = kwargs["init_mean"]
        init_params["std"] = kwargs["init_std"]
    elif init_params["dist"] == "eye":
        pass
    else:
        raise NotImplementedError()
    return init_params


if __name__ == '__main__':
    FLAGS = vars(get_args())
    print(FLAGS)
    main(**FLAGS)
