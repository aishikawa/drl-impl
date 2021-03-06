import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(Network, self).__init__()
        torch.manual_seed(seed)
        hidden1 = 64
        hidden2 = 64
        self.fc1 = nn.Linear(state_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class DuelingNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(DuelingNetwork, self).__init__()
        torch.manual_seed(seed)
        hidden1 = 64
        hidden2 = 64
        self.fc1 = nn.Linear(state_size, hidden1)

        self.vfc1 = nn.Linear(hidden1, hidden2)
        self.vfc2 = nn.Linear(hidden2, 1)

        self.afc1 = nn.Linear(hidden1, hidden2)
        self.afc2 = nn.Linear(hidden2, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)

        v = self.vfc1(x)
        v = F.relu(v)
        v = self.vfc2(v)

        a = self.afc1(x)
        a = F.relu(a)
        a = self.afc2(a)

        q = v + a - a.mean()

        return q
