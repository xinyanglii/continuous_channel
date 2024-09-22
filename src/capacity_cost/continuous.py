import logging
import math
from typing import Callable, Tuple

import numpy as np
import torch

from src.distributions import ContinuousChannelDistribution

log = logging.getLogger(__name__)
logprint = log.info
# logprint = print


class WassersteinGradientFlowContinuousOutput:
    def __init__(
        self,
        dist_channel: ContinuousChannelDistribution,
        num_samples_y: int,
        num_samples_y_eval: int,
        cost_func: Callable,
        cost_ub: float,
        init_x: torch.Tensor,
        optimizer_callable: Callable,
        scheduler_callable: Callable | None = None,
        lam: float = 0.0,
        lr_lam: float = 0.5,
        eps: float = 1e-4,
        max_iter: int = 1000,
        verbose: bool | int = True,
    ) -> None:
        """A Wasserstein gradient flow algorithm to computing the continuous input-
        output channel capacity.

        :param dist_channel: Distribution of channel p(y|x)
        :type dist_channel: ContinuousChannelDistribution
        :param num_samples_y: number of samples of channel output to perform the Monte Carlo integration
        :type num_samples_y: int
        :param num_samples_y_eval: number of samples of channel output to evaluate the capacity
        :type num_samples_y_eval: int
        :param cost_func: a function to compute the cost of the input x, input is x of shape [Nx, dim_x], output is a tensor of shape [Nx]
        :type cost_func: Callable
        :param cost_ub: cost uppperbound
        :type cost_ub: float
        :param init_x: initial particles of input x, shape: [Nx, dim_x]
        :type init_x: torch.Tensor
        :param optimizer_callable: a partial implementation of torch.optim.Optimizer, with only `params` argument to be passed in
        :type optimizer_callable: Callable
        :param scheduler_callable: a partial implementation of torch.optim.lr_scheduler.LRScheduler, with only `optimizer` argument to be passed in
                                    defaults to None
        :type scheduler_callable: Callable | None, optional
        :param lam: Lagrangian multiplier of the cost constraint, will be updated at each step according to dual update approach, defaults to 1.0
        :type lam: float, optional
        :param lr_lam: learning rate of lambda, defaults to 0.5
        :type lr_lam: float, optional
        :param eps: stopping criterion, defaults to 1e-4
        :type eps: float, optional
        :param max_iter: maximum number of iterations, defaults to 1000
        :type max_iter: int, optional
        :param verbose: if False, no logging information will be printed, otherwise, a int value to indicate the frequency of printing,
                        defaults to True
        :type verbose: bool | int, optional
        """

        self.dist_channel = dist_channel
        self.num_sample_y = num_samples_y
        self.num_sample_y_eval = num_samples_y_eval
        self.cost_func = cost_func

        self.x = init_x.clone()
        self.x.requires_grad = True
        self.optimizer = optimizer_callable([self.x])
        self.scheduler = None
        if scheduler_callable:
            self.scheduler = scheduler_callable(self.optimizer)

        self.lam = lam
        self.lr_lam = lr_lam
        self.cost_ub = cost_ub
        self.eps = eps
        self.max_iter = max_iter
        self.verbose = verbose

        self.logs: dict = {
            "rate": [],
            "cost": [],
            "objval": [],
            "ubval": [],
            "x": [],
        }

    def run(self) -> dict:
        lam = self.lam
        lr_lam = self.lr_lam

        for i in range(self.max_iter):
            self.optimizer.zero_grad()
            kld = self.dist_channel.compute_kld(self.num_sample_y, require_grad=True, x=self.x)

            cost_x = self.cost_func(self.x)

            with torch.no_grad():
                rate = kld.mean()
                cost = cost_x.mean()
            lam = lam + lr_lam * (cost - self.cost_ub)
            lam = max(0, lam)

            # Dx = kld - lam * (cost_x - self.cost_ub)
            Dx = kld - lam * cost_x
            # objval = Dx.detach().mean()
            objval = rate - lam * (cost - self.cost_ub)
            ubval = Dx.detach().max()

            self.logs["rate"].append(rate.item())
            self.logs["cost"].append(cost.item())
            self.logs["objval"].append(objval.item())
            self.logs["ubval"].append(ubval.item())

            Vx = -Dx
            Vx.backward(torch.ones_like(Vx))

            grad_x = self.x.grad.clone()
            if (grad_x**2).mean() < self.eps:
                if lr_lam == 0:
                    break
                elif torch.abs(cost - self.cost_ub) < self.eps:
                    break

            if self.verbose:
                if i % self.verbose == 0:
                    logprint(
                        f"Iter {i}: objval = {objval.item(): .4f}, rate = {rate.item(): .4f}, cost = {cost.item(): .4f}, lam = {lam: .4f}, lr = {self.optimizer.param_groups[0]['lr']: .6f}, grad_x = {(grad_x ** 2).mean(): .4f}",
                    )
            self.optimizer.step()

            if self.scheduler:
                try:
                    self.scheduler.step(objval)
                except Exception:
                    self.scheduler.step()

            self.logs["x"].append(self.x.numpy(force=True).copy())

        kld = self.dist_channel.compute_kld(self.num_sample_y_eval, require_grad=False, x=self.x)
        with torch.no_grad():
            rate = kld.mean()
            cost = self.cost_func(self.x).mean()
            objval = rate - lam * (cost - self.cost_ub)

        if self.verbose:
            logprint(
                f"Iter Final: objval = {objval.item(): .4f}, rate = {rate.item(): .4f}, cost = {cost.item(): .4f}",
            )

        self.logs["x"] = np.stack(self.logs["x"], axis=0)

        return self.logs
