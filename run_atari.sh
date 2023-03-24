#!/usr/bin/env bash

for env in 'MinAtar/Freeway-v1' 'MinAtar/SpaceInvaders-v1' 'MinAtar/Asterix-v1' 'MinAtar/Breakout-v1' 'MinAtar/Seaquest-v1'
do
    for ec_start in 0.5
    do
        for ec_end in 0.5
        do
            python train_atari.py --env_name $env --ec_factor_start $ec_start --ec_factor_end $ec_end --wandb --data_sharing &
            python train_atari.py --env_name $env --ec_factor_start $ec_start --ec_factor_end $ec_end --wandb &
        done
    done
done
