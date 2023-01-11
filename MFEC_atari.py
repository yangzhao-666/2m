import numpy as np
#np.random.seed(1)
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

import psutil

class MFECAgent():
    def __init__(self, buffer_size, discount, n_actions, config):
        self.memory = []
        self.n_actions = n_actions
        self.discount = discount
        self.config = config
        self.qec = QEC(n_actions, buffer_size, config)
        #self.rp_matrix = np.random.randn(np.prod(state_dim), random_projection_dim).astype(np.float32)

    def select_action(self, state):
        #projected_state = np.dot(state.flatten(), self.rp_matrix)
        state = hash(str(state))
        values = [np.inf for i in range(self.n_actions)]
        values = [
                self.qec.estimate(state, action)
                for action in range(self.n_actions)
                ]
        best_actions = np.argwhere(values == np.max(values)).flatten()
        action = np.random.choice(best_actions)

        return action

    def update(self, single_trajectory):
        value = 0.0
        for experience in reversed(single_trajectory):
            state = experience['state']
            state = hash(str(state))
            #projected_state = np.dot(state.flatten(), self.rp_matrix)
            value = value * self.discount + experience["reward"]
            action = experience['action']
            timestp = experience['timestp']
            self.qec.update(state, action, value, timestp)

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
            neighbor_idx = self._tree.query(state)[1][0][0]
            #print('neighbor idx: {} | len of buffer: {}'.format(neighbor_idx, len(self.states)))
            if np.allclose(self.states[neighbor_idx], state):
                return neighbor_idx
        return None

    def add(self, state, value, time):
        if len(self) < self.capacity:
            self.states.append(state)
            self.values.append(value)
            self.times.append(time)
        else:
            min_time_idx = int(np.argmin(self.times))
            if time > self.times[min_time_idx]:
                self.replace(state, value, time, min_time_idx)
        self._tree = KDTree(np.array(self.states).reshape(-1, 1))

    def replace(self, state, value, time, index):
        self.states[index] = state
        self.values[index] = value
        self.times[index] = time

    def __len__(self):
        return len(self.states)

class QEC(): #q values for episodic controller;
    def __init__(self, n_actions, buffer_size, config):
        self.buffers = tuple([ActionBuffer(buffer_size, config) for _ in range(n_actions)])
        self.config = config
        self.n_actions = n_actions

    def estimate(self, state, action):
        buffer = self.buffers[action]
        state = np.array(state).reshape(1, -1)
        state_index = buffer.find_states(state)

        if state_index:
            return buffer.values[state_index]

        # if we are dealing with Atari, we do not need neighbors for estimation since distance doesnt make sense in the hashed space.
        #return float('inf')
        return 0

    def update(self, state, action, value, time):
        buffer = self.buffers[action]
        #print('action: {}'.format(action))
        state = np.array(state).reshape(1, -1)
        state_index = buffer.find_states(state)
        if state_index is not None:
            max_value = max(buffer.values[state_index], value)
            max_time = max(buffer.times[state_index], time)
            buffer.replace(state, max_value, max_time, state_index)
        else:
            buffer.add(state, value, time)
