import logging
import os

import autorootcwd
import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from utils import mimoawgn_capacity, peak_cost, quad_cost, seed_everything

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="main.yaml")
def main(cfg: DictConfig):
    # hydra_cfg = HydraConfig.get()
    # print(OmegaConf.to_yaml(hydra_cfg))
    seed_everything(cfg.seed)

    if any(x in cfg.device.lower() for x in ["cuda", "gpu"]) and torch.cuda.is_available():
        gpu_id = HydraConfig.get().job.num % torch.cuda.device_count()
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    H = torch.randn(cfg.dim_y, cfg.dim_x)
    Q1, _ = torch.linalg.qr(H)
    H2 = torch.randn(cfg.dim_y, cfg.dim_x)
    Q2, _ = torch.linalg.qr(H2)
    eigvals = torch.rand(min(cfg.dim_y, cfg.dim_x))
    eigvals = (eigvals / eigvals.sum()) ** 0.5
    S = torch.zeros(cfg.dim_y, cfg.dim_x).diagonal_scatter(eigvals)
    channel_mat = Q1 @ S @ Q2.T
    # channel_mat = torch.randn(cfg.dim_y, cfg.dim_x)
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

    channel_mat = dist_channel.H_mean
    save_path = os.path.join(cfg.paths.output_dir, "logs.npz")
    np.savez(save_path, **logs, channel_mat=channel_mat.numpy(force=True))


if __name__ == "__main__":
    main()
