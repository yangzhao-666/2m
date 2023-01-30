#!/usr/bin/env bash

python train_emdqn.py --wandb --env_name 'MinAtar/Breakout-v1' --sticky_action_prob 0.2 &
python train_emdqn.py --wandb --env_name 'MinAtar/SpaceInvaders-v1' --sticky_action_prob 0.02 &
python train_emdqn.py --wandb --env_name 'MinAtar/Asterix-v1' --sticky_action_prob 0.1 &
python train_emdqn.py --wandb --env_name 'MinAtar/Freeway-v1' --sticky_action_prob 0 &
python train_emdqn.py --wandb --env_name 'MinAtar/Seaquest-v1' --sticky_action_prob 0.05 &
