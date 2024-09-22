#!/bin/bash

python src/fading.py experiment=fading_csir algorithm.lr_lam=5e-4 snr='range(-10, 6, 1)'

python src/fading.py experiment=fading_csir algorithm.lr_lam=1e-5 snr='range(6, 21, 1)'
