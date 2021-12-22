import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


def calc_accuracy(pred, label):
    max_index = pred.max(axis=1).indices
    return (max_index == label).sum().item() / label.shape[0]


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_regression(model, name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(-10, 10.1, 0.1)
    y = np.arange(-10, 10.1, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = model.predict(torch.tensor([X[i, j], Y[i, j]], dtype=torch.float))
    ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5, linewidth=0.3)
    ax.view_init(elev=60, azim=60)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.savefig(f"image/3dplot_{name}.png")


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def calc_angle(v1, v2):
    cos = (v1 * v2).sum(axis=1) / (torch.norm(v1, dim=1) * torch.norm(v2, dim=1) + 1e-12)
    cos = torch.clamp(cos, min=-1, max=1)
    acos = torch.acos(cos) * 180 / math.pi
    angle = 180 - torch.abs(acos - 180)
    return angle


def plot_hist_log(x, name):
    plt.figure()
    plt.hist(target_dist_plot[d].to('cpu').detach().numpy().copy(),
             bins=np.linspace(-6, 30, 40))
    plt.xscale('log')
    plt.xlabel('ratio')
    plt.ylabel('num')
    plt.savefig(name)
    plt.clf()
    plt.close()
