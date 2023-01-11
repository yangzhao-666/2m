import numpy as np

class TabularQAgent():
    def __init__(self, lr, n_actions=4, gamma=0.99):
        self.qec = [{} for _ in range(n_actions)]
        self.Qtable = {}
        self.lr = lr
        self.n_actions = n_actions
        self.gamma = gamma
        
    def update(self, batch_data):
        batch_s = batch_data.state
        batch_n_s = batch_data.n_state
        batch_a = batch_data.action
        batch_r = batch_data.reward
        batch_done = batch_data.done
        for s, n_s, a, r, d in zip(batch_s, batch_n_s, batch_a, batch_r, batch_done):
            if s not in self.Qtable.keys():
                self.Qtable[s] = [0 for _ in range(self.n_actions)]
            if n_s not in self.Qtable.keys():
                self.Qtable[n_s] = [0 for _ in range(self.n_actions)]
            td_target = r + (1 - d) * self.gamma * np.max(self.Qtable[n_s])
            td_error = td_target - self.Qtable[s][a]
            self.Qtable[s][a] = self.Qtable[s][a] + self.lr * td_error
            
    def select_action(self, s):
        values = [
            self._get_value(s, a) for a in range(self.n_actions)
        ]
        max_actions = np.where(values == np.max(values))[0]
        best_action = np.random.choice(max_actions)
        #print('EC values: {}'.format(values))
        return best_action
        
    def _get_value(self, s, action):
        if s not in self.Qtable.keys():
            return 0
        return self.Qtable[s][action]
