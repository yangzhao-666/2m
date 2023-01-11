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

from Env import FourRoomsEnv, ObsEnv, ImgEnv, GoalEnv
from RB import TwoMBuffer
from MFEC import MFECAgent
from TwoM import TwoMemoryAgent

from minigrid.wrappers import ReseedWrapper, FullyObsWrapper, RGBImgObsWrapper, ImgObsWrapper

import psutil

def evaluate(TwoM, eval_env):
    #canvas_agents = np.zeros((20, 20)) # -1 for EC, 1 for RL, 0 means the same(good/bad)
    returns = []
    trajs = np.zeros((20, 20))
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
            #trajs[s] += 1
        returns.append(eps_returns)

    return np.mean(returns), trajs
            
def train(TwoM, env, eval_env, config, wandb_session):
    #wandb_session.watch(rl.Q_net, log='all')
    i_steps = 0
    i_episode = 0
    #train_on_states = np.zeros((20, 20))
    #collected_data = {'ec': np.zeros((20, 20)), 'rl': np.zeros((20, 20))}
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
            n_s, r, done, _ = env.step(action)
            experience = {
                        'state': s,
                        'n_state': n_s,
                        'reward': r,
                        'done': done,
                        'action': action,
                        'timestp': i_steps
                }
            #collected_data[agent_str][s] += 1
            eps_returns += r
            #print(experience)
            single_trajectory.append(experience)
            s = n_s
            if TwoM.rb_ready():
                if i_steps % config.RL_train_freq == 0:
                    batch_data = TwoM.update_rl()
                    batch_state = batch_data.state
                    #for train_state in batch_state:
                    #    train_on_states[train_state] += 1
        TwoM.update_ec(single_trajectory)
        TwoM.collect_data(single_trajectory)
        TwoM.update_score(eps_returns)
        i_episode += 1
        if i_episode % config.eval_freq == 0:
            eval_return, eval_traj = evaluate(TwoM, eval_env)
            rl_score, ec_score = TwoM.get_score()
            epsilon = TwoM.calculate_eps()
            ec_factor = TwoM.calculate_ec_factor()
            sampling_weight = TwoM.calculate_sampling_weight()
            ec_size, ec_buffer_size, rl_buffer_size = TwoM.get_memory_size()
            wandb_session.log({'eval return': eval_return, 'rl score': rl_score, 'ec score': ec_score, 'eps': epsilon, 'ec size': ec_size, 'ec buffer size': ec_buffer_size, 'rl buffer size': rl_buffer_size, 'ec factor': ec_factor, 'sampling weight': sampling_weight, 'training steps': i_steps})
            print({'eval return': eval_return, 'rl score': rl_score, 'ec score': ec_score, 'eps': epsilon, 'ec size': ec_size, 'ec buffer size': ec_buffer_size, 'rl buffer size': rl_buffer_size, 'ec factor': ec_factor, 'sampling weight': sampling_weight, 'training steps': i_steps})
            '''
            plt.clf()
            ax = sns.heatmap(train_on_states, vmin=0, vmax=20000)
            wandb_session.log({'RL is trained on states': wandb.Image(ax)})
            plt.clf()
            ax = sns.heatmap(collected_data['rl'], vmin=0, vmax=20000)
            wandb_session.log({'RL Collected data distribution': wandb.Image(ax)})
            plt.clf()
            ax = sns.heatmap(collected_data['ec'], vmin=0, vmax=20000)
            wandb_session.log({'EC Collected data distribution': wandb.Image(ax)})
            plt.clf()
            ax = sns.heatmap(eval_traj, vmin=0, vmax=5)
            wandb_session.log({'2M eval trajectory': wandb.Image(ax)})
            '''

if __name__ == '__main__':
    torch.set_num_threads(10)
    description = '2MToyExample'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--project', type=str, default='2Mvector')
    #parser.add_argument('--env_name', type=str, default='MiniGrid-TwoRooms-v0')
    parser.add_argument('--env_name', type=str, default='LunarLander-v2')
    #parser.add_argument('--env_name', type=str, default='MinAtar/Breakout-v0')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--total_steps', type=int, default=500000)
    parser.add_argument('--eval_freq', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--target_update_freq', type=int, default=1000)
    parser.add_argument('--RL_train_freq', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--env_max_steps', type=int, default=100)

    parser.add_argument('--eps_start', type=float, default=0.9)
    parser.add_argument('--eps_end', type=float, default=0.1)
    parser.add_argument('--eps_decay_steps', type=int, default=100000)

    # do not start from 1 or 0, 1 means pure ec, 0 means pure rl.
    parser.add_argument('--ec_factor_start', type=float, default=0.9)
    parser.add_argument('--ec_factor_end', type=float, default=0.01)
    parser.add_argument('--ec_factor_decay_steps', type=int, default=100000)

    parser.add_argument('--sampling_weight_start', type=float, default=0.9)
    parser.add_argument('--sampling_weight_end', type=float, default=0.01)
    parser.add_argument('--sampling_weight_decay_steps', type=int, default=100000)

    parser.add_argument('--rb_capacity', type=int, default=50000)
    parser.add_argument('--rl_alg', type=str, default='DQN')
    parser.add_argument('--mfec_buffer_size', type=int, default=12500)
    parser.add_argument('--mfec_k', type=int, default=5)
    #parser.add_argument('--mfec_rp_dim', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--grid_size', type=int, default=20)

    args = parser.parse_args()
    for run in range(args.runs):
        if args.wandb:
            wandb_session = wandb.init(project=args.project, config=vars(args), name="run-%i"%(run), reinit=True, monitor_gym=False)
        else:
            wandb_session = wandb.init(project=args.project, config=vars(args), name="run-%i"%(run), reinit=True, mode='disabled')

        config = wandb.config

        '''
        env = gym.make(config.env_name)
        env = ReseedWrapper(FullyObsWrapper(env))
        env = FourRoomsEnv(env, max_steps=config.env_max_steps)

        eval_env = gym.make(config.env_name)
        eval_env = ReseedWrapper(FullyObsWrapper(eval_env))
        eval_env = FourRoomsEnv(eval_env, max_steps=config.env_max_steps)
        env = FullyObsWrapper(gym.make(config.env_name, size=config.grid_size))
        env = GoalEnv(env, max_steps=config.env_max_steps)

        eval_env = FullyObsWrapper(gym.make(config.env_name, size=config.grid_size))
        eval_env = GoalEnv(eval_env, max_steps=config.env_max_steps)
        '''
        from Env import NormalizedObs
        env = NormalizedObs(gym.make(config.env_name))
        eval_env = NormalizedObs(gym.make(config.env_name))

        #n_actions = env.action_space.n
        n_actions = 4
        input_dim = [8]

        if config.rl_alg == 'DQN':
            from DQN import DQNAgent
            rl = DQNAgent(gamma=config.gamma, target_update_freq=config.target_update_freq, input_dim=input_dim, n_actions=n_actions, hidden_dim=config.hidden_dim, lr=config.lr)
        elif config.rl_alg == 'DDQN':
            from DDQN import DDQNAgent
            rl = DDQNAgent(gamma=config.gamma, target_update_freq=config.target_update_freq, input_dim=input_dim, n_actions=n_actions, hidden_dim=config.hidden_dim, lr=config.lr)
        else: 
            raise NotImplementedError
        #ec = MFECAgent(buffer_size=config.mfec_buffer_size, k=config.mfec_k, n_actions=n_actions, config=config, discount=config.gamma, random_projection_dim=config.mfec_rp_dim, state_dim=input_dim)
        ec = MFECAgent(buffer_size=config.mfec_buffer_size, k=config.mfec_k, n_actions=n_actions, config=config, discount=config.gamma)
        TwoMrb = TwoMBuffer(capacity=config.rb_capacity, batch_size=config.batch_size)

        TwoMagent = TwoMemoryAgent(ec, rl, TwoMrb, n_actions, config)

        train(TwoMagent, env, eval_env, config, wandb_session)
