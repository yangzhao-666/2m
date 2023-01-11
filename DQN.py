import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQNAgent():
    def __init__(self, gamma, target_update_freq, input_dim, n_actions, hidden_dim=32, lr=0.01):
        self.gamma = gamma
        self.target_update_interval = target_update_freq
        self.n_actions = n_actions
        self.update_cntr = 0
        if len(input_dim) == 3:
            self.Q_net = QConvNet(input_dim[0], input_dim[1], input_dim[2], n_actions)
            self.Q_target_net = QConvNet(input_dim[0], input_dim[1], input_dim[2], n_actions)
        elif len(input_dim) == 1:
            self.Q_net = QLinearNet(input_dim[0], hidden_dim, n_actions)
            self.Q_target_net = QLinearNet(input_dim[0], hidden_dim, n_actions)
        self.target_hard_update()
        self.optimizer = optim.RMSprop(self.Q_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()

    def select_action(self, state):
        s = torch.from_numpy(np.array(state)).unsqueeze(0).float()
        action = self.Q_net(s).max(1)[1].view(1,1)
        return action

    def update(self, batch_data):
        self.update_cntr += 1
        batch_s = torch.FloatTensor(batch_data.state)
        batch_n_s = torch.FloatTensor(batch_data.n_state)
        batch_a = batch_data.action
        batch_r = torch.FloatTensor(batch_data.reward)
        batch_done = torch.FloatTensor(batch_data.done)
        # bootstrap for state that reaches maximum time step.
        try:
            batch_done[torch.logical_and(batch_r==-0.01, batch_done==1)] = torch.zeros(size=batch_done[torch.logical_and(batch_r==-0.01, batch_done==1)].shape)
        except:
            import ipdb; ipdb.set_trace()
        td_target = self.gamma * (1 - batch_done) * self.Q_target_net(batch_n_s).max(1)[0] + batch_r
        current_q = self.Q_net(batch_s).gather(1, torch.LongTensor(batch_a).unsqueeze(1))
        td_error = self.criterion(td_target, current_q.squeeze(1))
        self.optimizer.zero_grad()
        td_error.backward()
        self.optimizer.step()

        if self.update_cntr % self.target_update_interval == 0:
            self.target_hard_update()
        return td_error.item()
            
    def target_hard_update(self):
        for param, target_param in zip(self.Q_net.parameters(), self.Q_target_net.parameters()):
            target_param.data.copy_(param.data)
        
class QConvNet(nn.Module):
    def __init__(self, width, height, in_ch, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(width))
        convh = conv2d_size_out(conv2d_size_out(height))
        input_linear = convw * convh * 32
        self.out = nn.Linear(input_linear, n_actions)

    def forward(self, x):
        # NHWC -> NCHW
        x = x.permute(0, 3, 1, 2)# [0, 255] -> [0, 1]
        #x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.out(x.reshape(x.size(0), -1))
        return x

class QLinearNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
