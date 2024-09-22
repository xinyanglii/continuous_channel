#!/bin/bash

python src/mimoawgn.py experiment=mimoawgn

python src/mimoawgn.py experiment=mimoawgn dim_x=16 num_samples_x=64,128 algorithm.optimizer_callable.lr=5e-2 algorithm.lr_lam=1e-4 snr='range(-10, 6, 1)'

python src/mimoawgn.py experiment=mimoawgn dim_x=16 num_samples_x=64,128 algorithm.optimizer_callable.lr=5e-2 algorithm.lr_lam=1e-5 snr='range(6, 21, 1)'
