import numpy as np
from utils import cal_eps
from collections import deque

class TwoMemoryAgent():
    def __init__(self, ec, rl, rb, n_actions, config):
        self.ec = ec
        self.rl = rl
        self.rb = rb
        self.n_actions = n_actions
        self.ec_scores = deque(maxlen=50)
        self.rl_scores = deque(maxlen=50)
        self.current_agent = self.ec
        self.eps_s = config.eps_start
        self.eps_e = config.eps_end
        self.eps_d = config.eps_decay_steps
        self.ec_s = config.ec_factor_start
        self.ec_e = config.ec_factor_end
        self.ec_d = config.ec_factor_decay_steps
        self.sw_s = config.sampling_weight_start
        self.sw_e = config.sampling_weight_end
        self.sw_d = config.sampling_weight_decay_steps
        self.steps_cntr = 0
        self.config = config
        if 'MinAtar' in self.config.env_name and self.config.mm == False:
            from MFEC_atari import MFECAgent
        elif 'MinAtar' in self.config.env_name and self.config.mm == True:
            from MinMaxMFEC_atari import MFECAgent
        else:
            from MFEC import MFECAgent
        self.MFECAgent = MFECAgent
        self.data_sharing = config.data_sharing

    def choose_agent(self):
        ec_factor = self.calculate_ec_factor()
        if np.random.uniform() < ec_factor:
            self.current_agent = self.ec
        else:
            self.current_agent = self.rl
    
    def get_memory_size(self):
        # return size of (ec, ec_buffer, rl_buffer)
        #return [self.ec.get_size(), *self.rb.get_size()]
        return [self.ec.get_size(), self.rb.cntr]

    def choose_agent_eval(self):
        # for using pure rl and pure ec, we directly return corresponding agent
        if self.ec_s == 0:
            return self.rl, 'rl'
        elif self.ec_s == 1:
            return self.ec, 'ec'
        if np.mean(self.ec_scores) > np.mean(self.rl_scores):
            return self.ec, 'ec'
        else:
            return self.rl, 'rl'
    
    def get_current_agent_str(self):
        if isinstance(self.current_agent, self.MFECAgent):
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
        
    def calculate_sampling_weight(self):
        if self.sw_s == 0:
            return 0
        sampling_weight = cal_eps(self.sw_s, self.sw_e, self.sw_d, self.steps_cntr)
        return sampling_weight

    def calculate_ec_factor(self):
        if self.ec_s == 1 or self.ec_s == 0:
            return self.ec_s
        elif self.ec_s == -1: # meaning ec factor will be adaptively decided
            if np.sum(self.ec_scores) == 0 or np.sum(self.rl_scores) == 0: # if ec or rl has no progress, we should assign them 50%
                return 0.5
            s = np.sum(self.ec_scores) / np.sum(self.rl_scores)
            #ec_factor = s - 0.5
            ec_factor = 0.25 * s + 0.25
            ec_factor = np.clip(ec_factor, 0.05, 0.95)
            return ec_factor
        ec_factor = cal_eps(self.ec_s, self.ec_e, self.ec_d, self.steps_cntr)
        return ec_factor

    def collect_data(self, single_trajectory):
        if isinstance(self.current_agent, self.MFECAgent):
            #self.rb.ec_buffer.add(single_trajectory)
            if self.data_sharing:
                self.rb.add(single_trajectory, mark=1)
        else:
            #self.rb.rl_buffer.add(single_trajectory)
            self.rb.add(single_trajectory, mark=0)

    def update_score(self, score):
        if isinstance(self.current_agent, self.MFECAgent):
            self.ec_scores.append(score)
        else:
            self.rl_scores.append(score)
        
    def get_score(self):
        return np.mean(self.rl_scores), np.mean(self.ec_scores)

    def update_rl(self):
        #sampling_weight = self.calculate_sampling_weight()
        batch_data = self.rb.sample()
        self.rl.update(batch_data)
        return batch_data

    def update_ec(self, single_trajectory):
        if self.data_sharing:
            self.ec.update(single_trajectory)
        else:
            if isinstance(self.current_agent, self.MFECAgent):
                self.ec.update(single_trajectory)

    def rb_ready(self):
        '''
        if self.ec_s == 0 and self.rb.rl_buffer.cntr != 0 and self.sw_s == 0: # if full RL, and rl buffer is not empty, and data has to be sampled all from rl buffer
            return True
        elif self.ec_s == 0 and self.rb.rl_buffer.cntr != 0 and self.sw_s != 0: # if full RL, and rl buffer is not empty, and data has to be sampled all from rl buffer
            raise ValueError('You are tring use pure RL but still wanna sample data from ec buffer...')
        if self.rb.ec_buffer.cntr != 0 and self.rb.rl_buffer.cntr != 0:
            return True
        else:
            return False
        '''
        if self.rb.cntr > 0:
            return True

    def get_data_p(self):
        return self.rb.get_data_p()
