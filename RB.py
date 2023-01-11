import numpy as np
from collections import namedtuple

class ReplayBuffer():
    def __init__(self, capacity):
        self.rb = []
        self.capacity = capacity
        self.cntr = 0

    def sample(self, batch_size):
        idxes = np.random.choice(len(self.rb), batch_size)
        batch = []
        for experience in np.array(self.rb, dtype=object)[idxes]:
            batch.append((experience['state'], experience['action'], experience['reward'], experience['n_state'], experience['done']))
        return batch

    def add(self, single_trajectory):
        for experience in single_trajectory:
            self.rb.append(experience)
            self.cntr += 1
            if self.cntr >= self.capacity:
                self.cntr -= 1
                self.rb.pop(0)

class TwoMBuffer():
    def __init__(self, capacity, batch_size):
        self.rl_buffer = ReplayBuffer(capacity)
        self.ec_buffer = ReplayBuffer(capacity)
        self.batch_size = batch_size

    def sample(self, sampling_weight):
        rl_buffer_batch_size = int(self.batch_size * (1 - sampling_weight))
        ec_buffer_batch_size = self.batch_size - rl_buffer_batch_size
        Transition = namedtuple('Transition', ('state', 'action', 'reward', 'n_state', 'done'))
        rl_batch = self.rl_buffer.sample(rl_buffer_batch_size)
        ec_batch = self.ec_buffer.sample(ec_buffer_batch_size)
        rl_batch.extend(ec_batch)
        batch = []
        for experience in rl_batch:
            batch.append(Transition(experience[0], experience[1], experience[2], experience[3], experience[4]))
        batch = Transition(*zip(*batch))
        return batch

    def add(self, single_trajectory, who='rl'):
        if who == 'rl':
            self.rl_buffer.add(single_trajectory)
        elif who == 'ec':
            self.ec_buffer.add(single_trajectory)
        else:
            raise NotImplementedError

    def get_size(self):
        return len(self.ec_buffer.rb), len(self.rl_buffer.rb)
