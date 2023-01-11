#!/usr/bin/env bash

python train_atari.py --ec_factor_start 0 --sampling_weight_start 0 --wandb --rb_capacity 1000000 --eps_start 0.5 --eps_end 0.1 &
python train_atari.py --ec_factor_start 1 --sampling_weight_start 0 --wandb --mfec_buffer_size 250000 --eps_start 0.5 --eps_end 0.1 &

python train_atari.py --ec_factor_start 0.9 --ec_factor_end 0.1 --sampling_weight_start 0.1 --sampling_weight_end 0.9 --wandb --eps_start 0.5 --eps_end 0.1 &
python train_atari.py --ec_factor_start 0.9 --ec_factor_end 0.1 --sampling_weight_start 0.1 --sampling_weight_end 0.1 --wandb --eps_start 0.5 --eps_end 0.1 &
python train_atari.py --ec_factor_start 0.5 --ec_factor_end 0.1 --sampling_weight_start 0.1 --sampling_weight_end 0.5 --wandb --eps_start 0.5 --eps_end 0.1 &
python train_atari.py --ec_factor_start 0.5 --ec_factor_end 0.1 --sampling_weight_start 0.1 --sampling_weight_end 0.1 --wandb --eps_start 0.5 --eps_end 0.1 &
python train_atari.py --ec_factor_start 0.9 --ec_factor_end 0.1 --sampling_weight_start 0.9 --sampling_weight_end 0.1 --wandb --eps_start 0.5 --eps_end 0.1 &
python train_atari.py --ec_factor_start 0.9 --ec_factor_end 0.1 --sampling_weight_start 0.5 --sampling_weight_end 0.1 --wandb --eps_start 0.5 --eps_end 0.1 &
