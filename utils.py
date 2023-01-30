import numpy as np
import cv2

def post_explore(env, last_state, steps):
    pe_data = []
    i_steps = 0
    s = last_state
    info = {}
    while 'die' not in info.values() and i_steps < steps:
        action = env.action_space.sample()
        n_s, r, done, info = env.step(action)
        pe_data.append({
                                    'state': s, 
                                    'action': action, 
                                    'reward': 0, 
                                    'next_state': n_s, 
                                    'done': done})
        s = n_s
        i_steps += 1
    return pe_data, i_steps

def cal_eps(start, end, decay, i_steps):
    eps = end + (start - end) * np.exp(-1 * i_steps / decay)
    if start < end:
        return end - eps + start
    return eps

class ActionRescaler():
    def __init__(self, action_high, action_low):
        self.action_high = action_high
        self.action_low = action_low
        self.n = 1 / self.action_high

    def rescale_action(self, action):
    # rescale actions to range [-1, 1]
        scaled_action = action * self.n
        return scaled_action

    def scale_back_action(self, scaled_action):
        action = scaled_action / self.n
        return action

def softmax(x, tau=1):
    return np.exp(x/tau)/sum(np.exp(x/tau))

class DiscretizeSpace():
    def __init__(self, space, steps):
        self.steps = steps
        self.high = space.high
        self.low = space.low
        self.step_len = (self.high - self.low) / steps

    def get_index(self, continuous_state):
        # does not consider the boarder case, since it is almost not possible to reach the exact boundary
        continuous_state = np.clip(continuous_state, self.low, self.high - 0.00001)
        index = (continuous_state - self.low)// self.step_len
        return index.astype(int)

def normalize(min_v, max_v, v):
    return (v - min_v) / (max_v - min_v)

class RandomProjector():
    def __init__(self, from_size, to_size=32):
        self.rp_matrix = np.random.normal(size=(from_size, to_size))

    def project(self, state):
        projected_state = np.dot(state, self.rp_matrix)
        return projected_state

def dot(a,b):
    length = b.shape[1]
    c = np.zeros(length)
    for i in np.arange(length):
        c[i] = np.inner(a,b[:,i])
    return c
