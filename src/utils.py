import os
import random

import cvxpy as cp
import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def quad_cost(x):
    # x: [Nx, dim_x]
    power = (x**2).sum(dim=-1)
    return power


def peak_cost(x):
    power = (x**2).sum(dim=-1)
    return torch.ones_like(power) * power.max()


def mimoawgn_capacity(channel_mat, snr):
    # channel_mat: [dim_y, dim_x]
    dim_y, dim_x = channel_mat.shape
    snr_lin = 10 ** (snr / 10)
    if dim_x == 1:
        capacity = 0.5 * np.log2(1 + snr_lin * (channel_mat**2).sum())
        cov_x = snr_lin
    else:
        cov_x = cp.Variable((dim_x, dim_x), symmetric=True)
        constraints = [cov_x >> 0, cp.trace(cov_x) <= snr_lin]
        obj = cp.Maximize(0.5 * cp.log_det(np.eye(dim_y) + channel_mat @ cov_x @ channel_mat.T))
        prob = cp.Problem(obj, constraints)
        prob.solve()
        capacity = prob.value * np.log2(np.e)
        cov_x = cov_x.value

    return capacity, cov_x
