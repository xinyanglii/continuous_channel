import logging
import os

import autorootcwd
import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from utils import quad_cost, seed_everything

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="main.yaml")
def main(cfg: DictConfig):
    # hydra_cfg = HydraConfig.get()
    # print(OmegaConf.to_yaml(hydra_cfg))
    seed_everything(cfg.seed)

    if any(x in cfg.device.lower() for x in ["cuda", "gpu"]) and torch.cuda.is_available():
        try:
            gpu_id = HydraConfig.get().job.num % torch.cuda.device_count()
        except Exception:
            gpu_id = 0
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    channel_mat = torch.zeros(cfg.dim_y, cfg.dim_x)
    dist_channel = hydra.utils.instantiate(cfg.distribution, channel_mat=channel_mat.to(device))
    init_x = torch.rand(cfg.num_samples_x, cfg.dim_x) - 0.5
    cost_func = quad_cost
    cost_ub = 10 ** (cfg.snr / 10)

    init_x = init_x / (quad_cost(init_x).mean() ** 0.5) * (cost_ub**0.5)

    log.info(f"Cost upperbound: {cost_ub}")
    wfg = hydra.utils.instantiate(
        cfg.algorithm,
        dist_channel=dist_channel,
        cost_func=cost_func,
        cost_ub=cost_ub,
        init_x=init_x.to(device),
    )
    logs = wfg.run()

    cost = logs["cost"][-1]
    x_gauss = torch.randn(cfg.num_samples_x, cfg.dim_x, device=device) * (cost**0.5)
    capacity_lb = dist_channel.compute_kld(cfg.algorithm.num_samples_y_eval, x=x_gauss).mean()

    H_samples = dist_channel.dist_dH.sample((2048,)).to(device) + dist_channel.H_mean
    capacity_ub = 0.5 * torch.log(1 + H_samples**2 * cost).mean()

    log.info(f"Capacity lowerbound: {capacity_lb}, Capacity upperbound: {capacity_ub}")

    save_path = os.path.join(cfg.paths.output_dir, "logs.npz")
    np.savez(
        save_path,
        **logs,
        channel_mat=channel_mat.numpy(force=True),
        capacity_lb=capacity_lb.item(),
        capacity_ub=capacity_ub.item(),
    )


if __name__ == "__main__":
    main()
