import numpy as np
from collections import namedtuple

class ReplayBuffer():
    def __init__(self, capacity, batch_size):
        self.rb = []
        self.capacity = capacity
        self.batch_size = batch_size
        self.cntr = 0
        self.rb_where = []

    def sample(self):
        #
        Transition = namedtuple('Transition', ('state', 'action', 'reward', 'n_state', 'done'))
        idxes = np.random.choice(len(self.rb), self.batch_size)
        batch = []
        for experience in np.array(self.rb, dtype=object)[idxes]:
            #
            batch.append(Transition(experience['state'], experience['action'], experience['reward'], experience['n_state'], experience['done']))
        #
        batch = Transition(*zip(*batch))
        return batch

    def add(self, single_trajectory, flag):
        for experience in single_trajectory:
            self.rb.append(experience)
            self.rb_where.append(flag)
            self.cntr += 1
            if self.cntr >= self.capacity:
                self.cntr -= 1
                self.rb.pop(0)
                self.rb_where.pop(0)

class TwoMBuffer():
    def __init__(self, capacity, batch_size, sampling_weight=0.1):
        rl_buffer_batch_size = int(batch_size * (1 - sampling_weight))
        ec_buffer_batch_size = batch_size - rl_buffer_batch_size
        self.rl_buffer = ReplayBuffer(capacity, rl_buffer_batch_size)
        self.ec_buffer = ReplayBuffer(capacity, ec_buffer_batch_size)
        self.batch_size = batch_size

    def sample(self):
        Transition = namedtuple('Transition', ('state', 'action', 'reward', 'n_state', 'done'))
        rl_batch = self.rl_buffer.sample()
        ec_batch = self.ec_buffer.sample()
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

