#!/usr/bin/env bash

python train.py --ec_factor_start 0.99 --ec_factor_end 0.01 --sampling_weight_start 0.9 --sampling_weight_end 0.1 --wandb --ec_factor_decay_steps 2000000 &
python train.py --ec_factor_start 0.99 --ec_factor_end 0.01 --sampling_weight_start 0.1 --sampling_weight_end 0.9 --wandb --ec_factor_decay_steps 2000000 &
python train.py --ec_factor_start 0.99 --ec_factor_end 0.01 --sampling_weight_start 0.1 --sampling_weight_end 0.5 --wandb --ec_factor_decay_steps 2000000 &
python train.py --ec_factor_start 0.99 --ec_factor_end 0.01 --sampling_weight_start 0.5 --sampling_weight_end 0.1 --wandb --ec_factor_decay_steps 2000000 &
python train.py --ec_factor_start 0.99 --ec_factor_end 0.01 --sampling_weight_start 0.5 --sampling_weight_end 0.5 --wandb --ec_factor_decay_steps 2000000 &
python train.py --ec_factor_start 0.99 --ec_factor_end 0.01 --sampling_weight_start 0.1 --sampling_weight_end 0.1 --wandb --ec_factor_decay_steps 2000000 &
python train.py --ec_factor_start 0.99 --ec_factor_end 0.01 --sampling_weight_start 0.9 --sampling_weight_end 0.9 --wandb --ec_factor_decay_steps 2000000 &

python train.py --ec_factor_start 0.5 --ec_factor_end 0.01 --sampling_weight_start 0.9 --sampling_weight_end 0.1 --wandb --ec_factor_decay_steps 2000000 &
python train.py --ec_factor_start 0.5 --ec_factor_end 0.01 --sampling_weight_start 0.1 --sampling_weight_end 0.9 --wandb --ec_factor_decay_steps 2000000 &
python train.py --ec_factor_start 0.5 --ec_factor_end 0.01 --sampling_weight_start 0.1 --sampling_weight_end 0.5 --wandb --ec_factor_decay_steps 2000000 &
python train.py --ec_factor_start 0.5 --ec_factor_end 0.01 --sampling_weight_start 0.5 --sampling_weight_end 0.1 --wandb --ec_factor_decay_steps 2000000 &
python train.py --ec_factor_start 0.5 --ec_factor_end 0.01 --sampling_weight_start 0.5 --sampling_weight_end 0.5 --wandb --ec_factor_decay_steps 2000000 &
python train.py --ec_factor_start 0.5 --ec_factor_end 0.01 --sampling_weight_start 0.1 --sampling_weight_end 0.1 --wandb --ec_factor_decay_steps 2000000 &
python train.py --ec_factor_start 0.5 --ec_factor_end 0.01 --sampling_weight_start 0.9 --sampling_weight_end 0.9 --wandb --ec_factor_decay_steps 2000000 &
