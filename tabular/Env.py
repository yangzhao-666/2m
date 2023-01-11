import gymnasium as gym
import minigrid
from gym.spaces import Discrete

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
