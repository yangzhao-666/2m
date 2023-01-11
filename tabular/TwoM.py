import numpy as np
from utils import cal_eps
from collections import deque

from EC import ECAgent

class TwoMemoryAgent():
    def __init__(self, ec, rl, rb, n_actions, config):
        self.ec = ec
        self.rl = rl
        self.rb = rb
        self.n_actions = n_actions
        self.ec_scores = deque(maxlen=10)
        self.rl_scores = deque(maxlen=10)
        self.ec_factor = config.ec_factor
        self.current_agent = self.ec
        self.eps_s = config.eps_start
        self.eps_e = config.eps_end
        self.eps_d = config.eps_decay_steps
        self.steps_cntr = 0
        self.config = config

    def choose_agent(self):
        if np.random.uniform() < self.ec_factor:
            self.current_agent = self.ec
        else:
            self.current_agent = self.rl
    
    def choose_agent_eval(self):
        # for using pure rl and pure ec, we directly return corresponding agent
        if self.ec_factor == 0:
            return self.rl, 'rl'
        elif self.ec_factor == 1:
            return self.ec, 'ec'
        if np.mean(self.ec_scores) > np.mean(self.rl_scores):
            return self.ec, 'ec'
        else:
            return self.rl, 'rl'
    
    def get_current_agent_str(self):
        if isinstance(self.current_agent, ECAgent):
            return 'ec'
        else:
            return 'rl'

    def select_action(self, state, evaluate=False):
        self.steps_cntr += 1
        eps = self.calculate_eps()
        if not evaluate:
            if np.random.uniform() < eps:
                action = np.random.choice(range(self.n_actions))
            else:
                action = self.current_agent.select_action(state)
        else:
            action = self.current_agent.select_action(state)
        return action

    def calculate_eps(self):
        eps = cal_eps(self.eps_s, self.eps_e, self.eps_d, self.steps_cntr)
        return eps
        
    def collect_data(self, single_trajecotry):
        if isinstance(self.current_agent, ECAgent):
            self.rb.ec_buffer.add(single_trajecotry)
        else:
            self.rb.rl_buffer.add(single_trajecotry)

    def update_score(self, score):
        if isinstance(self.current_agent, ECAgent):
            self.ec_scores.append(score)
        else:
            self.rl_scores.append(score)
        
    def get_score(self):
        return np.mean(self.rl_scores), np.mean(self.ec_scores)

    def update_rl(self):
        batch_data = self.rb.sample()
        self.rl.update(batch_data)
        return batch_data

    def update_ec(self, single_trajecotry):
        self.ec.update(single_trajecotry)

    def rb_ready(self):
        if self.ec_factor == 0 and self.rb.rl_buffer.cntr != 0 and self.config.sampling_weight == 0: # if full RL, and rl buffer is not empty, and data has to be sampled all from rl buffer
            return True
        elif self.ec_factor == 0 and self.rb.rl_buffer.cntr != 0 and self.config.sampling_weight != 0: # if full RL, and rl buffer is not empty, and data has to be sampled all from rl buffer
            raise ValueError('You are tring use pure RL but still wanna sample data from ec buffer...')
        if self.rb.ec_buffer.cntr != 0 and self.rb.rl_buffer.cntr != 0:
            return True
        else:
            return False

    def get_Q_sum(self):
        rl_Q_sum = np.sum([*self.rl.Qtable.values()])
        ec_Q_sum = 0
        for qec in self.ec.qec:
            ec_Q_sum += np.sum([*qec.values()])
        return rl_Q_sum, ec_Q_sum
