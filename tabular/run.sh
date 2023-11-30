#!/usr/bin/env bash

python train_tab.py --ec_start 0.9 --ec_end 0.9 --wandb &
python train_tab.py --ec_start 0.5 --ec_end 0.5 --wandb &
python train_tab.py --ec_start 0.1 --ec_end 0.1 --wandb &
