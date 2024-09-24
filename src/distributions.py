import math

import torch
from einops import rearrange
from torch.distributions import Distribution, MultivariateNormal, Normal


class ContinuousDistribution:
    def __init__(self) -> None:
        pass

    def sample(self, num_samples: int, **conditions) -> torch.Tensor:
        raise NotImplementedError

    def log_prob(self, samples: torch.Tensor, **conditions) -> torch.Tensor:
        raise NotImplementedError

    def mclog_prob(self, num_samples: int, **conditions) -> torch.Tensor:
        """Monte Carlo integration to compute the log-likelihood of the samples given
        the conditions.

        That is, we first sample from the distribution and then compute the log-
        likelihood of the samples. Note that we don't involve the sampling step in the
        gradient computation, so we disable the gradient computation for it.
        """
        with torch.no_grad():
            samples = self.sample(num_samples, **conditions)
        return self.log_prob(samples, **conditions)


class ContinuousChannelDistribution(ContinuousDistribution):
    def __init__(self) -> None:
        super().__init__()

    def compute_kld(self, num_samples: int, require_grad: bool = False, **conditions) -> torch.Tensor:
        """Compute the KL divergence between P(Y|X) and P(Y) at the given conditions."""

        logp_y__x = self.mclog_prob(num_samples, **conditions)
        logp_y = torch.logsumexp(logp_y__x, dim=-1) - math.log(logp_y__x.shape[-1])
        logp_y__x = logp_y__x.diagonal(dim1=-2, dim2=-1)
        kld = logp_y__x - logp_y
        if require_grad:
            kld = kld * torch.exp(logp_y__x - logp_y__x.detach())

        return kld.mean(0)


class MultivariateGaussianDistribution(ContinuousDistribution):
    def __init__(self, mean: torch.Tensor, covariance: torch.Tensor) -> None:
        super().__init__()
        self.dist = MultivariateNormal(loc=mean, covariance_matrix=covariance)

    def sample(self, num_samples: int, **conditions) -> torch.Tensor:
        return self.dist.sample([num_samples])


# TODO: Complex Gaussian
class MIMOFadingAWGNDistribution(ContinuousChannelDistribution):
    def __init__(
        self,
        channel_mat: torch.Tensor,
        sigma_H: float = 0.0,
        sigma_noise: float = 1.0,
        num_samples_H=64,
        csi_rx: bool = False,
    ) -> None:
        """Distribution class of MIMO fading channel with AWGN noise.

        :param channel_mat: channel matrix, if sigma_H is set to nonzero, this is treated as the mean value of channel matrix,
                            shape: [num_ant_rx, num_ant_tx]
        :type channel_mat: torch.Tensor
        :param sigma_H: the standard deviation of channel matrix, if set to zero, channel matrix is fixed and there is no fading, defaults to 0.0
        :type sigma_H: float, optional
        :param sigma_noise: the standard deviation of AWGN noise, defaults to 1.0
        :type sigma_noise: float, optional
        :param num_samples_H: number of samples to randomly sample channel matrix in oder to sample the channel output y,
                            if no fading, this parameter is ommited and treated as 1, see the implementation below, defaults to 64
        :type num_samples_H: int, optional
        :param csi_rx: whether the receiver has the channel state information, defaults to False
        :type csi_rx: bool, optional
        """
        super().__init__()

        self.num_ant_rx, self.num_ant_tx = channel_mat.shape
        self.H_mean = channel_mat
        self.H_samples = channel_mat.unsqueeze(0)  # class property to store samples of channel matrix

        self.sigma_noise = sigma_noise
        self.num_samples_H = num_samples_H

        self.dist_noise = MultivariateNormal(
            loc=torch.zeros(self.num_ant_rx, device=channel_mat.device),
            covariance_matrix=sigma_noise * torch.eye(self.num_ant_rx, device=channel_mat.device),
        )

        if sigma_H:
            self.dist_dH = Normal(loc=0, scale=sigma_H)
        else:
            self.dist_dH = None

        self.csi_rx = csi_rx

    def sample(self, num_samples: int, **conditions) -> torch.Tensor:
        """Number of samples to randomly sample channel output for each condition['x']
        and each channel matrix sample according to y = p(y|x, H), i.e., y = Hx + n,
        where n is the noise, H is the channel matrix, x is the transmitted signal
        conditions should contain 'x' which is the transmitted signal, shape: [Nx,
        num_ant_tx] the other condition H is sampled internally and not needed to be
        provided."""
        x = conditions["x"]  # Nx, dim_x
        assert x.shape[1] == self.num_ant_tx

        if self.dist_dH:
            dH = self.dist_dH.sample([self.num_samples_H, self.num_ant_rx, self.num_ant_tx]).to(x.device)
            self.H_samples = self.H_mean.to(x.device) + dH  # Nh, dim_y, dim_x

        # because later we will perform backpropagation, to compute the gradient of log-likelihood, so we perform reparameterization trick
        # to disable gradient computation for the sampling process.
        with torch.no_grad():
            Hx = torch.einsum("...ij, ...j -> ...i", self.H_samples[None, ...], x[:, None, :])  # Nx, Nh, dim_y
            n = self.dist_noise.sample([num_samples, *Hx.shape[:-1]]).to(x.device)
            y = Hx[None, ...] + n  # num_samples, Nx, Nh, dim_y

        return y

    def log_prob(self, samples: torch.Tensor, **conditions) -> torch.Tensor:
        # return log p(y|x), we furst compute p(y|x, H) and then marginalize over H with Monte Carlo integration
        x = conditions["x"]
        Hx = torch.einsum("...ij, ...j -> ...i", self.H_samples[None, ...], x[:, None, :])  # Nx, Nh, dim_y
        if not self.csi_rx:
            n = samples[..., None, None, :] - Hx[None, None, ...]
            logp_y__x_H = self.dist_noise.log_prob(n)  # Ny, Nx, Nh, Nx, Nh
            logp_y__x = torch.logsumexp(logp_y__x_H, dim=-1) - math.log(len(self.H_samples))  # Ny, Nx, Nh, Nx
        else:
            # CSIR can be viewed as another channel output
            n = samples[..., None, :, :] - Hx
            logp_y__x = self.dist_noise.log_prob(n) + self.dist_dH.log_prob(self.H_samples - self.H_mean).sum(
                (-1, -2),
            )  # Ny, Nx, Nx, Nh
            logp_y__x = rearrange(logp_y__x, "Ny Nx1 Nx2 Nh -> Ny Nx1 Nh Nx2")

        return rearrange(logp_y__x, "Ny Nx1 Nh Nx2 -> (Ny Nh) Nx1 Nx2")
