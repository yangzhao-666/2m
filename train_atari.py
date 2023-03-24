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
from RB import ReplayBuffer
from TwoM import TwoMemoryAgent

def evaluate(TwoM, eval_env):
    returns = []
    agent, agent_str = TwoM.choose_agent_eval()
    for i in range(10):
        eps_returns = 0
        s = eval_env.reset()
        done = False
        while not done:
            action = agent.select_action(s)
            if isinstance(action, torch.Tensor):
                action = action.item()
            n_s, r, done, info = eval_env.step(action)
            s = n_s
            eps_returns += r
        returns.append(eps_returns)

    return np.mean(returns)
            
def train(TwoM, env, eval_env, config, wandb_session):
    #wandb_session.watch(rl.Q_net, log='all')
    i_steps = 0
    i_episode = 0
    steps_taken_exploration = 0
    while i_steps < config.total_steps:
        single_trajectory = []
        s = env.reset()
        TwoM.choose_agent()
        agent_str = TwoM.get_current_agent_str()
        eps_returns = 0
        done = False
        while not done:
            action = TwoM.select_action(s)
            if isinstance(action, torch.Tensor):
                action = action.item()
            i_steps += 1
            steps_taken_exploration += 1
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
            if TwoM.rb_ready():
                if i_steps % config.RL_train_freq == 0:
                    batch_data = TwoM.update_rl()
                    batch_state = batch_data.state
        TwoM.update_ec(single_trajectory)
        TwoM.collect_data(single_trajectory)
        TwoM.update_score(eps_returns)
        i_episode += 1
        if i_episode % config.eval_freq == 0:
            eval_return = evaluate(TwoM, eval_env)
            rl_score, ec_score = TwoM.get_score()
            epsilon = TwoM.calculate_eps()
            ec_factor = TwoM.calculate_ec_factor()
            sampling_weight = 0
            ec_size, rl_buffer_size = TwoM.get_memory_size()
            data_p = TwoM.get_data_p()
            steps_taken_exploration_per_episode = steps_taken_exploration / config.eval_freq
            wandb_session.log({'eval return': eval_return, 'rl score': rl_score, 'ec score': ec_score, 'eps': epsilon, 'ec size': ec_size, 'rl buffer size': rl_buffer_size, 'ec factor': ec_factor, 'sampling weight': sampling_weight, 'training steps': i_steps, 'data ec p': data_p, 'steps taken during data collection/exploration': steps_taken_exploration_per_episode})
            print({'eval return': eval_return, 'rl score': rl_score, 'ec score': ec_score, 'eps': epsilon, 'ec size': ec_size, 'rl buffer size': rl_buffer_size, 'ec factor': ec_factor, 'sampling weight': sampling_weight, 'training steps': i_steps, 'data ec p': data_p})
            steps_taken_exploration = 0

if __name__ == '__main__':
    torch.set_num_threads(10)
    description = '2MToyExample'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--project', type=str, default='2MMinAtarAllResults')
    #parser.add_argument('--env_name', type=str, default='MiniGrid-TwoRooms-v0')
    #parser.add_argument('--env_name', type=str, default='LunarLander-v2')
    parser.add_argument('--env_name', type=str, default='MinAtar/SpaceInvaders-v1')
    #parser.add_argument('--env_name', type=str, default='MinAtar/Breakout-v1')
    #parser.add_argument('--env_name', type=str, default='MinAtar/Asterix-v1')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--total_steps', type=int, default=3000000)
    parser.add_argument('--eval_freq', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--target_update_freq', type=int, default=1000)
    parser.add_argument('--RL_train_freq', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--env_max_steps', type=int, default=100)

    parser.add_argument('--eps_start', type=float, default=0.1)
    parser.add_argument('--eps_end', type=float, default=0.001)
    parser.add_argument('--eps_decay_steps', type=int, default=200000)

    # do not start from 1 or 0, 1 means pure ec, 0 means pure rl.
    parser.add_argument('--ec_factor_start', type=float, default=0.9)
    parser.add_argument('--ec_factor_end', type=float, default=0.1)
    parser.add_argument('--ec_factor_decay_steps', type=int, default=500000)

    parser.add_argument('--sampling_weight_start', type=float, default=0.9)
    parser.add_argument('--sampling_weight_end', type=float, default=0.01)
    parser.add_argument('--sampling_weight_decay_steps', type=int, default=500000)

    parser.add_argument('--total_memory_size', type=int, default=100000)
    parser.add_argument('--rb_capacity', type=int, default=250000) # this will be calculated according to the total memory size
    parser.add_argument('--rl_alg', type=str, default='DQN')
    parser.add_argument('--mfec_buffer_size', type=int, default=125000) # this will be calculated according to the total memory size
    parser.add_argument('--mfec_k', type=int, default=3)
    parser.add_argument('--mfec_rp_dim', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--grid_size', type=int, default=20)

    parser.add_argument('--sticky_action_prob', type=float, default=0)

    parser.add_argument('--data_sharing', default=False, action='store_true')

    args = parser.parse_args()

    if 'SpaceInvaders' in args.env_name:
        n_actions = 4
        input_dim = [10, 10, 6]
        args.sticky_action_prob = 0.02
    elif 'Freeway' in args.env_name:
        n_actions = 3
        input_dim = [10, 10, 7]
        #args.eval_freq = 10
        args.sticky_action_prob = 0.01
    elif 'Asterix' in args.env_name:
        n_actions = 5
        input_dim = [10, 10, 4]
        args.sticky_action_prob = 0.1
    elif 'Breakout' in args.env_name:
        n_actions = 3
        input_dim = [10, 10, 4]
        args.sticky_action_prob = 0.2
    elif 'Seaquest' in args.env_name:
        n_actions = 6
        input_dim = [10, 10, 10]
        args.sticky_action_prob = 0.05
    else:
        raise ValueError('Plz specify action space and input space for the env {}'.format(str(config.env_name)))

    # calculate size of different buffers
    if args.ec_factor_start == 1:
        # pure ec
        args.mfec_buffer_size = args.total_memory_size / n_actions
        args.rb_capacity = 1
        n_rb = 0
    elif args.ec_factor_start == 0:
        # pure rl
        args.mfec_buffer_size = 1
        args.rb_capacity = args.total_memory_size
        n_rb = 1
    else:
        args.mfec_buffer_size = args.total_memory_size / (2*n_actions)
        args.rb_capacity = args.total_memory_size / 2
        n_rb = 1

    print('total memory size: {} | mfec size: {} * {} | rb capacity: {} * {}'.format(args.total_memory_size, args.mfec_buffer_size, n_actions, args.rb_capacity, n_rb))

    for run in range(args.runs):
        if args.wandb:
            wandb_session = wandb.init(project=args.project, config=vars(args), name="run-%i"%(run), reinit=True, monitor_gym=False)
        else:
            wandb_session = wandb.init(project=args.project, config=vars(args), name="run-%i"%(run), reinit=True, mode='disabled')

        config = wandb.config

        env = gym.make(config.env_name, sticky_action_prob=config.sticky_action_prob)
        eval_env = gym.make(config.env_name, sticky_action_prob=config.sticky_action_prob)

        if config.rl_alg == 'DQN':
            from DQN import DQNAgent
            rl = DQNAgent(gamma=config.gamma, target_update_freq=config.target_update_freq, input_dim=input_dim, n_actions=n_actions, hidden_dim=config.hidden_dim, lr=config.lr)
        else: 
            raise NotImplementedError

        ec = MFECAgent(buffer_size=config.mfec_buffer_size, k=config.mfec_k, n_actions=n_actions, config=config, discount=config.gamma, random_projection_dim=config.mfec_rp_dim, state_dim=input_dim)
        rb = ReplayBuffer(capacity=config.rb_capacity, batch_size=config.batch_size)

        TwoMagent = TwoMemoryAgent(ec, rl, rb, n_actions, config)

        train(TwoMagent, env, eval_env, config, wandb_session)
