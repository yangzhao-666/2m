import gymnasium as gym
import minigrid
from gym.spaces import Discrete
from gym import spaces

import numpy as np

class FourRoomsEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, max_steps=200):
        super().__init__(env)
        self.action_space = Discrete(4, )
        self.steps_taken = 0
        self.max_steps = max_steps

    def reset(self):
        self.steps_taken = 0
        original_reset = self.env.reset()
        agent_pos = np.where(original_reset[0]['image'][:,:,0]==10)
        direction = original_reset[0]['direction']
#obs = (agent_pos[0][0], agent_pos[1][0], direction)
        obs = (agent_pos[0][0], agent_pos[1][0])
        return obs
    def step(self, action):
        # action: 0 up, 1 down, 2 left, 3 right.
        if self.env.agent_dir == 0: # >
            if action == 0:
                simulated_step = self.env.step(0)
                original_step = self.env.step(2)
            elif action == 1:
                simulated_step = self.env.step(1)
                original_step = self.env.step(2)
            elif action == 2:
                simulated_step = self.env.step(0)
                simulated_step = self.env.step(0)
                original_step = self.env.step(2)
            elif action == 3:
                original_step = self.env.step(2)
        elif self.env.agent_dir == 1: # down
            if action == 0:
                simulated_step = self.env.step(0)
                simulated_step = self.env.step(0)
                original_step = self.env.step(2)
            elif action == 1:
                original_step = self.env.step(2)
            elif action == 2:
                simulated_step = self.env.step(1)
                original_step = self.env.step(2)
            elif action == 3:
                simulated_step = self.env.step(0)
                original_step = self.env.step(2)
        elif self.env.agent_dir == 2: # <
            if action == 0:
                simulated_step = self.env.step(1)
                original_step = self.env.step(2)
            elif action == 1:
                simulated_step = self.env.step(0)
                original_step = self.env.step(2)
            elif action == 2:
                original_step = self.env.step(2)
            elif action == 3:
                simulated_step = self.env.step(0)
                simulated_step = self.env.step(0)
                original_step = self.env.step(2)
        elif self.env.agent_dir == 3: # up
            if action == 0:
                original_step = self.env.step(2)
            elif action == 1:
                simulated_step = self.env.step(0)
                simulated_step = self.env.step(0)
                original_step = self.env.step(2)
            elif action == 2:
                simulated_step = self.env.step(0)
                original_step = self.env.step(2)
            elif action == 3:
                simulated_step = self.env.step(1)
                original_step = self.env.step(2)
                
        self.steps_taken += 1
        agent_pos = np.where(original_step[0]['image'][:,:,0]==10)
        direction = original_step[0]['direction']
        #obs = (agent_pos[0][0], agent_pos[1][0], direction)
        obs = (agent_pos[0][0], agent_pos[1][0])
        reward = original_step[1]
        done = False
        if reward == 0:
            reward = -0.01
        else:
            reward = 1
        if reward != -0.01:
            done = True
        info = {}
        if self.steps_taken >= self.max_steps:
            done = True
        return obs, reward, done, info

class ObsEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, max_steps=200):
        super().__init__(env)
        self.steps_taken = 0
        self.max_steps = max_steps

    def reset(self):
        self.steps_taken = 0
        original_reset = self.env.reset()
        obs = original_reset[0]['image'][:,:,0].flatten()
        return obs

    def step(self, action):
        self.steps_taken += 1

        original_step = self.env.step(action)
        obs = original_step[0]['image'][:,:,0].flatten()
        reward = original_step[1]
        done = False
        if reward == 0:
            reward = -0.01
        else:
            reward = 1
        if reward != -0.01:
            done = True
        info = {}
        if self.steps_taken >= self.max_steps:
            done = True
        return obs, reward, done, info

class ImgEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, max_steps=200):
        super().__init__(env)
        self.steps_taken = 0
        self.max_steps = max_steps

    def reset(self):
        self.steps_taken = 0
        original_reset = self.env.reset()
        obs = original_reset[0]['image']
        return obs

    def step(self, action):
        self.steps_taken += 1

        original_step = self.env.step(action)
        obs = original_step[0]['image']
        reward = original_step[1]
        done = False
        if reward == 0:
            reward = -0.01
        else:
            reward = 1
        if reward != -0.01:
            done = True
        info = {}
        if self.steps_taken >= self.max_steps:
            done = True
        return obs, reward, done, info

class GoalEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, max_steps=200):
        super().__init__(env)
        self.action_space = Discrete(4, )
        self.steps_taken = 0
        self.max_steps = max_steps

    def reset(self):
        self.steps_taken = 0
        original_reset = self.env.reset()
        agent_pos = np.where(original_reset[0]['image'][:,:,0]==10)
        direction = original_reset[0]['direction']
        goal_pos = np.where(original_reset[0]['image'][:,:,0]==8)
#obs = (agent_pos[0][0], agent_pos[1][0], direction)
        obs = (agent_pos[0][0], agent_pos[1][0], goal_pos[0][0], goal_pos[1][0])
        return obs
    def step(self, action):
        # action: 0 up, 1 down, 2 left, 3 right.
        if self.env.agent_dir == 0: # >
            if action == 0:
                simulated_step = self.env.step(0)
                original_step = self.env.step(2)
            elif action == 1:
                simulated_step = self.env.step(1)
                original_step = self.env.step(2)
            elif action == 2:
                simulated_step = self.env.step(0)
                simulated_step = self.env.step(0)
                original_step = self.env.step(2)
            elif action == 3:
                original_step = self.env.step(2)
        elif self.env.agent_dir == 1: # down
            if action == 0:
                simulated_step = self.env.step(0)
                simulated_step = self.env.step(0)
                original_step = self.env.step(2)
            elif action == 1:
                original_step = self.env.step(2)
            elif action == 2:
                simulated_step = self.env.step(1)
                original_step = self.env.step(2)
            elif action == 3:
                simulated_step = self.env.step(0)
                original_step = self.env.step(2)
        elif self.env.agent_dir == 2: # <
            if action == 0:
                simulated_step = self.env.step(1)
                original_step = self.env.step(2)
            elif action == 1:
                simulated_step = self.env.step(0)
                original_step = self.env.step(2)
            elif action == 2:
                original_step = self.env.step(2)
            elif action == 3:
                simulated_step = self.env.step(0)
                simulated_step = self.env.step(0)
                original_step = self.env.step(2)
        elif self.env.agent_dir == 3: # up
            if action == 0:
                original_step = self.env.step(2)
            elif action == 1:
                simulated_step = self.env.step(0)
                simulated_step = self.env.step(0)
                original_step = self.env.step(2)
            elif action == 2:
                simulated_step = self.env.step(0)
                original_step = self.env.step(2)
            elif action == 3:
                simulated_step = self.env.step(1)
                original_step = self.env.step(2)
                
        self.steps_taken += 1
        agent_pos = np.where(original_step[0]['image'][:,:,0]==10)
        direction = original_step[0]['direction']
        goal_pos = np.where(original_step[0]['image'][:,:,0]==8)
        if not goal_pos[0]:
            goal_pos = agent_pos
        #obs = (agent_pos[0][0], agent_pos[1][0], direction)
        obs = (agent_pos[0][0], agent_pos[1][0], goal_pos[0][0], goal_pos[1][0])
        reward = original_step[1]
        done = False
        if reward == 0:
            reward = -0.01
        else:
            reward = 1
        if reward != -0.01:
            done = True
        info = {}
        if self.steps_taken >= self.max_steps:
            done = True
        return obs, reward, done, info

class FlattenObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(low=self.observation_space.low.flatten(), high=self.observation_space.high.flatten())

    def reset(self):
        original_reset = self.env.reset()
        return original_reset.flatten()
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = obs.flatten()
        return obs, reward, done, info

from utils import normalize
import gym
class NormalizedObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.min_v = np.array((1.5, 1.5, 5, 5, 3.14, 5, 1, 1))
        self.max_v = np.array((-1.5, -1.5, -5, -5, -3.14, -5, 0, 0))
        self.observation_space = spaces.Box(low=np.zeros(self.observation_space.low.shape), high=np.ones(self.observation_space.high.shape))
    
    def observation(self, obs):
        normal_obs = normalize(self.min_v, self.max_v, obs)
        return normal_obs
