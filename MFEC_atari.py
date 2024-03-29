import numpy as np
#np.random.seed(1)
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from utils import dot

import psutil

class MFECAgent():
    def __init__(self, buffer_size, k, discount, n_actions, state_dim, random_projection_dim, config):
        self.memory = []
        self.n_actions = n_actions
        self.discount = discount
        self.config = config
        self.qec = QEC(n_actions, buffer_size, k, config)
        self.rp_matrix = np.random.randn(np.prod(state_dim), random_projection_dim).astype(np.float32)

    def select_action(self, state):
        projected_state = dot(state.flatten(), self.rp_matrix)
        values = [np.inf for i in range(self.n_actions)]
        values = [
                self.qec.estimate(projected_state, action)
                for action in range(self.n_actions)
                ]
        best_actions = np.argwhere(values == np.max(values)).flatten()
        action = np.random.choice(best_actions)

        return action

    def estimate_values(self, batch_state, batch_action): 
        values = []
        for s, a in zip(batch_state, batch_action):
            projected_state = dot(s.flatten(), self.rp_matrix)
            values.append(self.qec.estimate(projected_state, a))
        return values

    def update(self, single_trajectory):
        value = 0.0
        for experience in reversed(single_trajectory):
            state = experience['state']
            projected_state = dot(state.flatten(), self.rp_matrix)
            #projected_state = np.dot(state.flatten(), self.rp_matrix)
            value = value * self.discount + experience["reward"]
            action = experience['action']
            timestp = experience['timestp']
            self.qec.update(projected_state, action, value, timestp)

    def save(self, results_dir):
        with open(os.path.join(results_dir, "agent.pkl"), "wb") as file:
            pickle.dump(self, file, 2)

    def load(path):
        with open(path, "rb") as file:
            return pickle.load(file)

    def get_size(self):
        size = np.sum([len(self.qec.buffers[a].times) for a in range(self.n_actions)])
        return size

class ActionBuffer():
    def __init__(self, capacity, config):
        self._tree = None
        self.capacity = capacity
        self.states = []
        self.values = []
        self.times = []
        self.config = config

    def find_states(self, state):
        if self._tree:
            neighbor_idx = self._tree.query([state])[1][0][0]
            #print('neighbor idx: {} | len of buffer: {}'.format(neighbor_idx, len(self.states)))
            if np.allclose(self.states[neighbor_idx], state):
                return neighbor_idx
        return None

    def find_neighbors(self, state, k):
        return self._tree.query([state], k)[1][0] if self._tree else []

    def add(self, state, value, time):
        if len(self) < self.capacity:
            self.states.append(state)
            self.values.append(value)
            self.times.append(time)
        else:
            min_time_idx = int(np.argmin(self.times))
            if time > self.times[min_time_idx]:
                self.replace(state, value, time, min_time_idx)
        self._tree = KDTree(np.array(self.states))

    def replace(self, state, value, time, index):
        self.states[index] = state
        self.values[index] = value
        self.times[index] = time

    def __len__(self):
        return len(self.states)

class QEC(): #q values for episodic controller;
    def __init__(self, n_actions, buffer_size, k, config):
        self.buffers = tuple([ActionBuffer(buffer_size, config) for _ in range(n_actions)])
        self.k = k
        self.config = config
        self.n_actions = n_actions

    def estimate(self, state, action):
        buffer = self.buffers[action]
        state_index = buffer.find_states(state)

        if state_index:
            return buffer.values[state_index]

        if len(buffer) <= self.k:
            # if there are not enough neighbors under the given action, this action will be taken. 
            return float('inf')

        value = 0.0
        if self.k == 0:
            return value
        neighbors = buffer.find_neighbors(state, self.k)
        for neighbor in neighbors:
            value += buffer.values[neighbor]
        return value / self.k

    def update(self, state, action, value, time):
        buffer = self.buffers[action]
        #print('action: {}'.format(action))
        state_index = buffer.find_states(state)
        if state_index is not None:
            max_value = max(buffer.values[state_index], value)
            max_time = max(buffer.times[state_index], time)
            buffer.replace(state, max_value, max_time, state_index)
        else:
            buffer.add(state, value, time)
