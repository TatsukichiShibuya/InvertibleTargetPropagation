from utils import plot_regression, worker_init_fn
from dataset import make_regression_dataset, make_classification_dataset

from bp_net import bp_net
from dttp_net import dttp_net
from mytp_net import mytp_net

import os
import sys
import torch
import argparse
import numpy as np
from torch import nn

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'


def get_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--problem",    type=str, default="regression")
    parser.add_argument("--datasize",   type=int, default=1000)

    # model architecture
    parser.add_argument("--depth",      type=int, default=4)
    parser.add_argument("--in_dim",     type=int, default=2)
    parser.add_argument("--hid_dim",    type=int, default=100)
    parser.add_argument("--out_dim",    type=int, default=1)
    parser.add_argument("--activation_function", type=str, default="leakyrelu")

    # learning algorithm
    parser.add_argument("--algorithm",  type=str, default="BP")
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--batchsize",  type=int, default=100)
    parser.add_argument("--seed",       type=int, default=39)

    # parameters used in BP
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-6)

    # parameters used in DTTP
    parser.add_argument("--stepsize",   type=float, default=2e-5)
    parser.add_argument("--learning_rate_for_backward", "-lrb", type=float, default=1e-2)
    parser.add_argument("--direct_depth", type=int, default=2)
    parser.add_argument("--b_epochs",   type=int, default=0)
    parser.add_argument("--sigma",      type=float, default=0.01)

    args = parser.parse_args()
    return args


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(**kwargs):
    # initialize
    fix_seed(kwargs["seed"])

    # make dataset
    if kwargs["problem"] == "regression":
        trainset, validset, testset = make_regression_dataset(kwargs["datasize"], kwargs["in_dim"])
        loss_function = nn.MSELoss(reduction="sum")
    elif kwargs["problem"] == "classification":
        trainset, validset, testset = make_classification_dataset(kwargs["datasize"])
        loss_function = nn.CrossEntropyLoss(reduction="sum")

    else:
        sys.tracebacklimit = 0
        raise NotImplementedError(f"dataset : {kwargs['dataset']} ?")
    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=kwargs["batchsize"],
                                               shuffle=True,
                                               num_workers=2,
                                               pin_memory=True,
                                               worker_init_fn=worker_init_fn)
    valid_loader = torch.utils.data.DataLoader(validset,
                                               batch_size=kwargs["batchsize"],
                                               shuffle=False,
                                               num_workers=2,
                                               pin_memory=True,
                                               worker_init_fn=worker_init_fn)
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=kwargs["batchsize"],
                                              shuffle=False,
                                              num_workers=2,
                                              pin_memory=True,
                                              worker_init_fn=worker_init_fn)

    # initialize model
    if kwargs["algorithm"] == "BP":
        model = bp_net(depth=kwargs["depth"],
                       in_dim=kwargs["in_dim"],
                       out_dim=kwargs["out_dim"],
                       hid_dim=kwargs["hid_dim"],
                       activation_function=kwargs["activation_function"],
                       loss_function=loss_function)
    elif kwargs["algorithm"] == "DTTP":
        model = dttp_net(depth=kwargs["depth"],
                         in_dim=kwargs["in_dim"],
                         out_dim=kwargs["out_dim"],
                         hid_dim=kwargs["hid_dim"],
                         direct_depth=kwargs["direct_depth"],
                         activation_function=kwargs["activation_function"],
                         loss_function=loss_function)
    elif kwargs["algorithm"] == "MyTP":
        model = mytp_net(depth=kwargs["depth"],
                         in_dim=kwargs["in_dim"],
                         out_dim=kwargs["out_dim"],
                         hid_dim=kwargs["hid_dim"],
                         direct_depth=kwargs["direct_depth"],
                         activation_function=kwargs["activation_function"],
                         loss_function=loss_function)
    else:
        sys.tracebacklimit = 0
        raise NotImplementedError(f"algorithm : {kwargs['algorithm']} ?")

    # train
    if kwargs["algorithm"] == "BP":
        model.train(train_loader, valid_loader, kwargs["epochs"], kwargs["learning_rate"])
    elif kwargs["algorithm"] == "DTTP":
        model.train(train_loader, valid_loader, kwargs["epochs"], kwargs["stepsize"],
                    kwargs["learning_rate_for_backward"], kwargs["b_epochs"], kwargs["sigma"])
    elif kwargs["algorithm"] == "MyTP":
        model.train(train_loader, valid_loader, kwargs["epochs"], kwargs["stepsize"],
                    kwargs["learning_rate_for_backward"], kwargs["b_epochs"], kwargs["sigma"])

    # test
    print(f"\ttest  : {model.test(test_loader)}")

    # plot
    if kwargs["problem"] == "regression" and kwargs["in_dim"] == 2:
        plot_regression(model, kwargs["algorithm"])


if __name__ == '__main__':
    FLAGS = vars(get_args())
    print(FLAGS)
    main(**FLAGS)
