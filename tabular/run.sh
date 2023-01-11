#!/usr/bin/env bash

python train.py --ec_factor 0 --sampling_weight 0 --total_steps 200000 --wandb &

python train.py --ec_factor 1 --sampling_weight 0 --total_steps 200000 --wandb &

python train.py --total_steps 200000 --wandb &
