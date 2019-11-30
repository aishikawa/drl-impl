import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, state_size, action_size):
        super(Network, self).__init__()
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
