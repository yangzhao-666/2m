#!/usr/bin/env bash

python train_lunar_lander.py --ec_factor_start 0 --sampling_weight_start 0 --wandb &
python train_lunar_lander.py --ec_factor_start 1 --sampling_weight_start 0 --wandb &

python train_lunar_lander.py --ec_factor_start 0.9 --ec_factor_end 0.1 --sampling_weight_start 0.1 --sampling_weight_end 0.9 --wandb &
python train_lunar_lander.py --ec_factor_start 0.5 --ec_factor_end 0.1 --sampling_weight_start 0.1 --sampling_weight_end 0.5 --wandb &
python train_lunar_lander.py --ec_factor_start 0.9 --ec_factor_end 0.1 --sampling_weight_start 0.9 --sampling_weight_end 0.1 --wandb &
python train_lunar_lander.py --ec_factor_start 0.9 --ec_factor_end 0.1 --sampling_weight_start 0.5 --sampling_weight_end 0.1 --wandb &
