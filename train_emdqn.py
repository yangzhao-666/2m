import numpy as np
import time
import wandb
#import gymnasium as gym
import gym
import argparse
from collections import deque
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from MFEC_atari import MFECAgent
from EMDQN import EMDQNAgent
from RB import ReplayBuffer

from utils import cal_eps

import psutil

def evaluate(emdqn, eval_env):
    returns = []
    for i in range(10):
        eps_returns = 0
        s = eval_env.reset()
        done = False
        while not done:
            action = emdqn.select_action(s)
            if isinstance(action, torch.Tensor):
                action = action.item()
            n_s, r, done, info = eval_env.step(action)
            s = n_s
            eps_returns += r
        returns.append(eps_returns)

    return np.mean(returns)
            
def train(emdqn, rb, env, eval_env, config, wandb_session):
    i_steps = 0
    i_episode = 0
    while i_steps < config.total_steps:
        single_trajectory = []
        s = env.reset()
        eps_returns = 0
        done = False
        while not done:
            eps = cal_eps(config.eps_start, config.eps_end, config.eps_decay_steps, i_steps)
            if np.random.uniform() < eps:
                action = np.random.choice(range(emdqn.n_actions))
            else:
                action = emdqn.select_action(s)
            if isinstance(action, torch.Tensor):
                action = action.item()
            i_steps += 1
            n_s, r, done, _ = env.step(action)
            experience = {
                        'state': s,
                        'n_state': n_s,
                        'reward': r,
                        'done': done,
                        'action': action,
                        'timestp': i_steps
                }
            eps_returns += r
            single_trajectory.append(experience)
            s = n_s
            if rb.cntr > config.batch_size:
                if i_steps % config.RL_train_freq == 0:
                    batch_data = rb.sample()
                    emdqn.update(batch_data)
        emdqn.em.update(single_trajectory)
        rb.add(single_trajectory, mark=0)
        i_episode += 1
        if i_episode % config.eval_freq == 0:
            eval_return = evaluate(emdqn, eval_env)
            ec_size = emdqn.em.get_size()
            rl_buffer_size = rb.cntr
            wandb_session.log({'eval return': eval_return, 'eps': eps, 'ec size': ec_size, 'rl buffer size': rl_buffer_size, 'training steps': i_steps})
            print({'eval return': eval_return, 'eps': eps, 'ec size': ec_size, 'rl buffer size': rl_buffer_size, 'training steps': i_steps})

if __name__ == '__main__':
    torch.set_num_threads(10)
    description = 'EMDQN'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--project', type=str, default='2MAtariSto1SEEDwithSticky')
    parser.add_argument('--env_name', type=str, default='MinAtar/SpaceInvaders-v1')
    parser.add_argument('--rl_alg', type=str, default='emdqn')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--total_steps', type=int, default=3000000)
    parser.add_argument('--eval_freq', type=int, default=500)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--target_update_freq', type=int, default=1000)
    parser.add_argument('--RL_train_freq', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--env_max_steps', type=int, default=100)

    parser.add_argument('--eps_start', type=float, default=0.1)
    parser.add_argument('--eps_end', type=float, default=0.001)
    parser.add_argument('--eps_decay_steps', type=int, default=200000)

    parser.add_argument('--total_memory_size', type=int, default=100000)
    parser.add_argument('--rb_capacity', type=int, default=250000) # this will be calculated according to the total memory size
    parser.add_argument('--mfec_buffer_size', type=int, default=125000) # this will be calculated according to the total memory size
    parser.add_argument('--mfec_k', type=int, default=3)
    parser.add_argument('--mfec_rp_dim', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--grid_size', type=int, default=20)

    parser.add_argument('--sticky_action_prob', type=float, default=0)

    parser.add_argument('--emdqn_lambda', type=float, default=0.01)

    args = parser.parse_args()

    if 'SpaceInvaders' in args.env_name:
        n_actions = 4
        input_dim = [10, 10, 6]
        #args.sticky_action_prob = 0.02
    elif 'Freeway' in args.env_name:
        n_actions = 3
        input_dim = [10, 10, 7]
        args.eval_freq = 10
        #args.sticky_action_prob = 0
    elif 'Asterix' in args.env_name:
        n_actions = 5
        input_dim = [10, 10, 4]
        #args.sticky_action_prob = 0.1
    elif 'Breakout' in args.env_name:
        n_actions = 3
        input_dim = [10, 10, 4]
        #args.sticky_action_prob = 0.2
    elif 'Seaquest' in args.env_name:
        n_actions = 6
        input_dim = [10, 10, 10]
        #args.sticky_action_prob = 0.05
    else:
        raise ValueError('Plz specify action space and input space for the env {}'.format(str(config.env_name)))

    # calculate size of different buffers
    args.mfec_buffer_size = args.total_memory_size / (2*n_actions)
    args.rb_capacity = args.total_memory_size / 2

    print('total memory size: {} | mfec size: {} * {} | rb capacity: {}'.format(args.total_memory_size, args.mfec_buffer_size, n_actions, args.rb_capacity))

    for run in range(args.runs):
        if args.wandb:
            wandb_session = wandb.init(project=args.project, config=vars(args), name="run-%i"%(run), reinit=True, monitor_gym=False)
        else:
            wandb_session = wandb.init(project=args.project, config=vars(args), name="run-%i"%(run), reinit=True, mode='disabled')

        config = wandb.config

        env = gym.make(config.env_name, sticky_action_prob=config.sticky_action_prob)
        eval_env = gym.make(config.env_name, sticky_action_prob=config.sticky_action_prob)

        rb = ReplayBuffer(config.rb_capacity, batch_size=args.batch_size)
        em = MFECAgent(buffer_size=config.mfec_buffer_size, k=config.mfec_k, n_actions=n_actions, config=config, discount=config.gamma, random_projection_dim=config.mfec_rp_dim, state_dim=input_dim)
        emdqn = EMDQNAgent(em, gamma=config.gamma, target_update_freq=config.target_update_freq, input_dim=input_dim, n_actions=n_actions, hidden_dim=config.hidden_dim, lr=config.lr, em_lambda=config.emdqn_lambda)
        train(emdqn, rb, env, eval_env, config, wandb_session)
