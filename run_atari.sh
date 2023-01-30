#!/usr/bin/env bash

for env in 'MinAtar/SpaceInvaders-v1'
do
    for sp in 0.02
    do
        python train_atari.py --ec_factor_start 1 --sampling_weight_start 0 --wandb --env_name $env --sticky_action_prob $sp &
        python train_atari.py --ec_factor_start 0 --sampling_weight_start 0 --wandb --env_name $env --sticky_action_prob $sp &
        python train_atari.py --ec_factor_start 0.9 --ec_factor_end 0.1 --sampling_weight_start 0.9 --sampling_weight_end 0.9 --wandb --env_name $env --sticky_action_prob $sp &
        python train_atari.py --ec_factor_start 0.9 --ec_factor_end 0.1 --sampling_weight_start 0.9 --sampling_weight_end 0.1 --wandb --env_name $env --sticky_action_prob $sp &
        python train_atari.py --ec_factor_start 0.9 --ec_factor_end 0.1 --sampling_weight_start 0.1 --sampling_weight_end 0.1 --wandb --env_name $env --sticky_action_prob $sp &
        python train_atari.py --ec_factor_start 0.5 --ec_factor_end 0.1 --sampling_weight_start 0.5 --sampling_weight_end 0.5 --wandb --env_name $env --sticky_action_prob $sp &
        python train_atari.py --ec_factor_start 0.5 --ec_factor_end 0.1 --sampling_weight_start 0.1 --sampling_weight_end 0.1 --wandb --env_name $env --sticky_action_prob $sp &
    done
done
