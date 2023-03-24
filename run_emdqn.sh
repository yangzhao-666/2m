#!/usr/bin/env bash

python train_emdqn.py --wandb --env_name 'MinAtar/Freeway-v1' --sticky_action_prob 0.01 &
