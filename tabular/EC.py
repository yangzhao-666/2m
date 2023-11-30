import numpy as np

class ECAgent():
    def __init__(self, n_actions=4, gamma=0.99):
        self.qec = [{} for _ in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        
    def update(self, single_trajectory):
        value = 0
        for experience in reversed(single_trajectory):
            value = experience['reward'] + self.gamma * value
            #print(value)
            s = experience['state']
            if s not in self.qec[experience['action']].keys():
                self.qec[experience['action']][s] = value
                #print('adding...')
            else:
                self.qec[experience['action']][s] = np.max((self.qec[experience['action']][s], value))
                #print('updating...')
            
    def select_action(self, s):
        values = [
            self._get_value(s, a) for a in range(self.n_actions)
        ]
        max_actions = np.where(values == np.max(values))[0]
        best_action = np.random.choice(max_actions)
        #print('EC values: {}'.format(values))
        return best_action
        
    def _get_value(self, s, action):
        if s not in self.qec[action].keys():
            return 0
        return self.qec[action][s]
