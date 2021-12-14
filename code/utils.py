import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def calc_accuracy(pred, label):
    max_index = pred.max(axis=1).indices
    return (max_index == label).sum().item() / label.shape[0]


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
