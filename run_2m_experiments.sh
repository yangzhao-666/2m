#!/usr/bin/env bash

python train_2m.py --env_name 'MinAtar/Breakout-v1' --ec_start 0.5 --ec_end 0.1 --data_sharing --wandb &
python train_2m.py --env_name 'MinAtar/SpaceInvaders-v1' --ec_start 0.9 --ec_end 0.1 --wandb &
python train_2m.py --env_name 'MinAtar/Asterix-v1' --ec_start 0.9 --ec_end 0.1 --wandb &
python train_2m.py --env_name 'MinAtar/Seaquest-v1' --ec_start 0.1 --ec_end 0.1 --data_sharing --wandb &
python train_2m.py --env_name 'MinAtar/Freeway-v1' --ec_start 0.5 --ec_end 0.1 --data_sharing --wandb &
