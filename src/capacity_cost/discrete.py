import logging
import math

import torch
from torch.nn.functional import kl_div

from src.distributions import ContinuousDistribution

log = logging.getLogger(__name__)
# logprint = log.info
logprint = print


def blahut_arimoto_step(logp_x: torch.Tensor, Dx: torch.Tensor, tau: float) -> torch.Tensor:
    # logp_x: [Nx], Dx: [Nx]
    logp_x = logp_x + tau * Dx
    logp_x = logp_x - torch.logsumexp(logp_x, dim=0)
    return logp_x


class BlahutArimotoDiscreteOutput:
    def __init__(
        self,
        dist_channel: torch.Tensor,
        cost_x: torch.Tensor | None = None,
        cost_ub: float = 1.0,
        init_p_x: torch.Tensor | None = None,
        init_tau: float = 1.0,
        lam: float = 0,
        lr_lam: float = 0.5,
        eps: float = 1e-4,
        max_iter: int = 1000,
        accelerate: bool = False,
        verbose: bool | int = True,
    ) -> None:
        """classical and accelerated Blahut-Arimoto algorithm for discrete output channel, see
        https://en.wikipedia.org/wiki/Blahut%E2%80%93Arimoto_algorithm
        Matz, Gerald, and Pierre Duhamel. "Information geometric formulation and
            interpretation of accelerated Blahut-Arimoto-type algorithms." Information theory workshop. IEEE, 2004.

        :param dist_channel: a matrix of size [Ny, Nx], where Ny is the number of output symbols, Nx is the number of input symbols
                             indicating the channel distribution p(y|x)
        :type dist_channel: torch.Tensor
        :param cost_x: a vector of cost for each x, or set to none indicating no cost constraint, defaults to None
        :type cost_x: torch.Tensor | None, optional
        :param cost_ub: the cost upperbound, defaults to 1.0
        :type cost_ub: float, optional
        :param init_p_x: initial p(x), defaults to None
        :type init_p_x: torch.Tensor | None, optional
        :param init_tau: initial step size of BA, if the parameter accelerate set to True, tau will be updated at each set,
                            defaults to 1.0
        :type init_tau: float, optional
        :param lam: the Lagrangian multiplier lambda for the cost constraint, this value will be updated according to dual ascent approach,
                    defaults to 0
        :type lam: float, optional
        :param lr_lam: the learning rate used to update lambda, defaults to 0.5
        :type lr_lam: float, optional
        :param eps: stopping criterion for BA, defaults to 1e-4
        :type eps: float, optional
        :param max_iter: maximum number of iterations, defaults to 1000
        :type max_iter: int, optional
        :param accelerate: if True, accelerated version of BA is performed, if False, the classical one, defaults to False
        :type accelerate: bool, optional
        :param verbose: if False, no logging information will be printed, otherwise, a int value to indicate the frequency of printing,
                        defaults to True
        :type verbose: bool | int, optional
        """
        assert torch.all(dist_channel >= 0)
        assert torch.allclose(dist_channel.sum(0), torch.tensor(1.0))
        assert dist_channel.ndim == 2

        self.logp_y__x = torch.log(dist_channel)

        self.card_x = dist_channel.shape[1]
        if init_p_x is None:
            init_p_x = torch.ones(self.card_x) / self.card_x
        assert len(init_p_x) == self.card_x
        assert torch.all(init_p_x >= 0) and torch.allclose(init_p_x.sum(), torch.tensor(1.0))
        self.logp_x = torch.log(init_p_x)
        self.init_tau = init_tau

        if cost_x is None:
            cost_x = torch.zeros(self.card_x)

        self.cost_ub = cost_ub

        assert len(cost_x) == self.card_x
        self.cost_x = cost_x
        self.lam = lam
        self.lr_lam = lr_lam
        self.eps = eps
        self.max_iter = max_iter
        self.verbose = verbose
        self.accelerate = accelerate

        self.logs: dict = {
            "rate": [],
            "cost": [],
            "objval": [],
            "ubval": [],
            "logp_x_optimal": None,
        }

    def run(self) -> dict:
        tau = self.init_tau
        lam = self.lam
        lr_lam = self.lr_lam

        logp_y = torch.logsumexp(self.logp_y__x + self.logp_x, dim=1)

        for i in range(self.max_iter):
            kld = kl_div(self.logp_y__x, logp_y[:, None], reduction="none", log_target=True).sum(0)

            rate = (kld * torch.exp(self.logp_x)).sum()
            cost = (self.cost_x * torch.exp(self.logp_x)).sum()

            lam = lam + lr_lam * (cost - self.cost_ub)
            lam = max(0, lam)

            Dx = kld - lam * self.cost_x
            objval = rate - lam * cost
            ubval = Dx.max()

            self.logs["rate"].append(rate.item())
            self.logs["cost"].append(cost.item())
            self.logs["objval"].append(objval.item())
            self.logs["ubval"].append(ubval.item())

            if self.verbose:
                if i % self.verbose == 0:
                    logprint(
                        f"Iter {i}: objval = {objval.item()}, ubval = {ubval.item()}, rate = {rate.item()}, cost = {cost.item()}, tau = {tau}",
                    )

            if torch.nan_to_num(ubval - objval) < self.eps:
                break

            last_logp_x = self.logp_x.clone()
            last_logp_y = logp_y.clone()
            self.logp_x = blahut_arimoto_step(self.logp_x, Dx, tau)
            logp_y = torch.logsumexp(self.logp_y__x + self.logp_x, dim=1)

            if self.accelerate:
                kld_y = kl_div(logp_y, last_logp_y, reduction="sum", log_target=True)
                kld_x = kl_div(self.logp_x, last_logp_x, reduction="sum", log_target=True)
                # tau = kld_y / kld_x
                kld_y_x = kld_x / kld_y
                if kld_y_x >= 0:
                    tau = kld_y_x

        if self.verbose:
            logprint(
                f"Final: objval = {objval.item()}, ubval = {ubval.item()}, rate = {rate.item()}, cost = {cost.item()}",
            )

        self.logs["logp_x_optimal"] = self.logp_x.clone()

        return self.logs


class BlahutArimotoContinuousOutput:
    def __init__(
        self,
        dist_channel: ContinuousDistribution,
        support_x: torch.Tensor,
        num_samples_y: int = 64,
        cost_x: torch.Tensor | None = None,
        cost_ub: float = 1.0,
        init_p_x: torch.Tensor | None = None,
        init_tau: float = 1.0,
        lam: float = 0,
        lr_lam: float = 0.5,
        eps: float = 1e-4,
        max_iter: int = 1000,
        accelerate: bool = False,
        verbose: bool | int = True,
    ) -> None:
        """The classicla and accelerated Blahut-Arimoto algorithm for continuous channel
        distribution, here we perform the Monte Carlo integration to sample and compute
        p(y|x) and p(y)."""

        self.dist_channel = dist_channel
        self.support_x = support_x
        self.num_samples_y = num_samples_y

        self.card_x = support_x.shape[0]
        if init_p_x is None:
            init_p_x = torch.ones(self.card_x) / self.card_x
        assert len(init_p_x) == self.card_x
        assert torch.all(init_p_x >= 0) and torch.allclose(init_p_x.sum(), torch.tensor(1.0))
        self.logp_x = torch.log(init_p_x)
        self.init_tau = init_tau

        if cost_x is None:
            cost_x = torch.zeros(self.card_x)

        self.cost_ub = cost_ub

        assert len(cost_x) == self.card_x
        self.cost_x = cost_x
        self.lam = lam
        self.lr_lam = lr_lam
        self.eps = eps
        self.max_iter = max_iter
        self.verbose = verbose
        self.accelerate = accelerate

        self.logs: dict = {
            "rate": [],
            "cost": [],
            "objval": [],
            "ubval": [],
            "optimal_logp_x": None,
        }

    def run(self) -> dict:
        tau = self.init_tau
        lam = self.lam

        for i in range(self.max_iter):
            logp_y__x = self.dist_channel.mclog_prob(self.num_samples_y, x=self.support_x)
            logp_y = torch.logsumexp(logp_y__x + self.logp_x, dim=-1)

            kld = (logp_y__x.diagonal(dim1=-2, dim2=-1) - logp_y).mean(0)

            rate = (kld * torch.exp(self.logp_x)).sum()
            cost = (self.cost_x * torch.exp(self.logp_x)).sum()

            # lam = lam + self.lr_lam * (cost - self.cost_ub)
            lam = (1 - self.lr_lam) * lam + self.lr_lam / (cost + 1e-4) * (cost - self.cost_ub)

            Dx = kld - lam * self.cost_x

            objval = rate - lam * cost
            ubval = Dx.max()

            self.logs["rate"].append(rate.item())
            self.logs["cost"].append(cost.item())
            self.logs["objval"].append(objval.item())
            self.logs["ubval"].append(ubval.item())

            if self.verbose:
                if i % self.verbose == 0:
                    logprint(
                        f"Iter {i}: objval = {objval.item()}, ubval = {ubval.item()}, rate = {rate.item()}, cost = {cost.item()}, tau = {tau}",
                    )

            if torch.nan_to_num(ubval - objval) < self.eps:
                break

            last_logp_x = self.logp_x.clone()
            last_logp_y = logp_y.clone()
            self.logp_x = blahut_arimoto_step(self.logp_x, Dx, tau)
            logp_y = torch.logsumexp(logp_y__x + self.logp_x, dim=-1)

            if self.accelerate:
                kld_y = kl_div(logp_y, last_logp_y, reduction="sum", log_target=True)
                kld_x = kl_div(self.logp_x, last_logp_x, reduction="sum", log_target=True)
                # tau = kld_y / kld_x
                kld_y_x = kld_x / kld_y
                if kld_y_x >= 0:
                    tau = kld_y_x

        if self.verbose:
            logprint(
                f"Final: objval = {objval.item()}, ubval = {ubval.item()}, rate = {rate.item()}, cost = {cost.item()}",
            )

        self.logs["logp_x_optimal"] = self.logp_x

        return self.logs
