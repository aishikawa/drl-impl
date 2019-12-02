from network import Network
from replay_buffer import ReplayBuffer
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


class DqnAgent:
    def __init__(self, state_size: int, action_size: int, batch_size=64, gamma=0.99, target_update_every=100, seed=1):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_every = target_update_every

        np.random.seed(seed)
        self.num_learn = 0

        self.q_network = Network(state_size, action_size, seed)
        self.target_network = Network(state_size, action_size, seed)

        self.replay_memory = ReplayBuffer(buffer_size=10000, seed=seed)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-4)

    def step(self, state, action, reward, next_state, done):
        self.replay_memory.add(state, action, reward, next_state, done)
        if len(self.replay_memory) >= self.batch_size:
            self.learn()

    def act(self, state, epsilon):
        if np.random.rand() > epsilon:
            state = torch.from_numpy(state).float().unsqueeze(0)
            self.q_network.eval()
            with torch.no_grad():
                action_values = self.q_network(state)
            self.q_network.train()
            action = np.argmax(action_values.data.numpy())
        else:
            action = np.random.randint(self.action_size)
        return action

    def learn(self):
        batch = self.replay_memory.sample(self.batch_size)
        states, actions, rewards, n_states, dones = (torch.from_numpy(v) for v in batch)
        states = states.float()
        rewards = rewards.float()
        n_states = n_states.float()
        dones = dones.float()

        max_next_state_q = self.target_network(n_states).detach().max(1)[0].unsqueeze(1)
        target = rewards + (self.gamma * max_next_state_q * (1 - dones))
        q = self.q_network(states).gather(1, actions)
        loss = F.mse_loss(q, target)
        loss.backward()
        self.optimizer.step()

        if self.num_learn % self.target_update_every:
            self.target_update()

        self.num_learn += 1

    def target_update(self):
        self.target_network.load_state_dict(self.q_network.state_dict())