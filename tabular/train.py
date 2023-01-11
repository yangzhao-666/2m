import numpy as np
import time
import wandb
import gymnasium as gym
#import gym
import argparse
from collections import deque
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from Env import FourRoomsEnv, ObsEnv
from RB import TwoMBuffer
from EC import ECAgent
from TabularQ import TabularQAgent
from TwoM import TwoMemoryAgent

from minigrid.wrappers import ReseedWrapper, FullyObsWrapper, RGBImgObsWrapper, ImgObsWrapper

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
            n_s, r, done, info = eval_env.step(action)
            s = n_s
            eps_returns += r
            trajs[s] += 1
        returns.append(eps_returns)

    return np.mean(returns), trajs
            
def train(TwoM, env, eval_env, config, wandb_session):
    #wandb_session.watch(rl.Q_net, log='all')
    i_steps = 0
    i_episode = 0
    train_on_states = np.zeros((20, 20))
    collected_data = {'ec': np.zeros((20, 20)), 'rl': np.zeros((20, 20))}
    while i_steps < config.total_steps:
        single_trajectory = []
        s = env.reset()
        TwoM.choose_agent()
        agent_str = TwoM.get_current_agent_str()
        eps_returns = 0
        done = False
        while not done:
            action = TwoM.select_action(s)
            i_steps += 1
            #print('s_g: {} | action: {}'.format(s_g, action))
            n_s, r, done, _ = env.step(action)
            experience = {
                        'state': s,
                        'n_state': n_s,
                        'reward': r,
                        'done': done,
                        'action': action
                }
            collected_data[agent_str][s] += 1
            eps_returns += r
            single_trajectory.append(experience)
            s = n_s
            if TwoM.rb_ready():
                if i_steps % config.RL_train_freq == 0:
                    batch_data = TwoM.update_rl()
                    batch_state = batch_data.state
                    for train_state in batch_state:
                        train_on_states[train_state] += 1
        TwoM.update_ec(single_trajectory)
        TwoM.collect_data(single_trajectory)
        TwoM.update_score(eps_returns)
        i_episode += 1
        if i_episode % config.eval_freq == 0:
            eval_return, eval_traj = evaluate(TwoM, eval_env)
            rl_score, ec_score = TwoM.get_score()
            wandb_session.log({'eval return': eval_return, 'rl score': rl_score, 'ec score': ec_score, 'training steps': i_steps})
            epsilon = TwoM.calculate_eps()
            print({'eval return': eval_return, 'training steps': i_steps, 'epsilon': epsilon})
            plt.clf()
            ax = sns.heatmap(train_on_states, vmin=0, vmax=5000)
            wandb_session.log({'RL is trained on states': wandb.Image(ax)})
            plt.clf()
            ax = sns.heatmap(collected_data['rl'], vmin=0, vmax=5000)
            wandb_session.log({'RL Collected data distribution': wandb.Image(ax)})
            plt.clf()
            ax = sns.heatmap(collected_data['ec'], vmin=0, vmax=5000)
            wandb_session.log({'EC Collected data distribution': wandb.Image(ax)})
            plt.clf()
            ax = sns.heatmap(eval_traj, vmin=0, vmax=5)
            wandb_session.log({'2M eval trajectory': wandb.Image(ax)})
            rl_Q_sum, ec_Q_sum = TwoM.get_Q_sum()
            wandb_session.log({'rl Q sum': rl_Q_sum, 'ec Q sum':ec_Q_sum, 'training steps': i_steps})
        return

if __name__ == '__main__':
    torch.set_num_threads(5)
    description = '2MToyExample'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--project', type=str, default='2Mtabular')
    parser.add_argument('--env_name', type=str, default='MiniGrid-TwoRooms-v0')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--total_steps', type=int, default=50000)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--eps_start', type=float, default=0.9)
    parser.add_argument('--eps_end', type=float, default=0.1)
    parser.add_argument('--eps_decay_steps', type=int, default=20000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--RL_train_freq', type=int, default=8)
    parser.add_argument('--env_max_steps', type=int, default=100)
    parser.add_argument('--ec_factor', type=float, default=0.5)
    parser.add_argument('--rb_capacity', type=int, default=50000)
    parser.add_argument('--sampling_weight', type=float, default=0.8)
    parser.add_argument('--mfec_buffer_size', type=int, default=50000)
    parser.add_argument('--mfec_k', type=int, default=3)
    parser.add_argument('--env_seed', type=int, default=0)

    args = parser.parse_args()
    for run in range(args.runs):
        if args.wandb:
            wandb_session = wandb.init(project=args.project, config=vars(args), name="run-%i"%(run), reinit=True, monitor_gym=False)
        else:
            wandb_session = wandb.init(project=args.project, config=vars(args), name="run-%i"%(run), reinit=True, mode='disabled')

        config = wandb.config

        env = gym.make(config.env_name)
        env = ReseedWrapper(FullyObsWrapper(env), seeds=[config.env_seed])
        env = FourRoomsEnv(env, max_steps=config.env_max_steps)

        eval_env = gym.make(config.env_name)
        eval_env = ReseedWrapper(FullyObsWrapper(eval_env), seeds=[config.env_seed])
        eval_env = FourRoomsEnv(eval_env, max_steps=config.env_max_steps)

        #n_actions = env.action_space.n
        n_actions = 4

        rl = TabularQAgent(config.lr, n_actions, config.gamma)
        ec = ECAgent(n_actions, config.gamma)
        TwoMrb = TwoMBuffer(capacity=config.rb_capacity, batch_size=32, sampling_weight=config.sampling_weight)

        TwoMagent = TwoMemoryAgent(ec, rl, TwoMrb, n_actions, config)

        train(TwoMagent, env, eval_env, config, wandb_session)
