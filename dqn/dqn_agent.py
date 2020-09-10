from dqn.network import Network, DuelingNetwork
from dqn.replay_buffer import ReplayBuffer
import numpy as np
import torch
import torch.optim as optim


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DqnAgent:
    def __init__(self, state_size: int, action_size: int,
                 batch_size=64, gamma=0.99,
                 soft_target_update=False, target_update_every=100, soft_update_ratio=0.001,
                 double=False,
                 duel=False,
                 pr_alpha=0.5,
                 pr_initial_beta=0.4,
                 pr_max_beta=1.0,
                 multi_step=3,
                 seed=1):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.is_soft_target_update = soft_target_update
        self.target_update_every = target_update_every
        self.soft_update_ratio = soft_update_ratio
        self.double = double

        self.num_learn = 0

        if duel:
            self.q_network = DuelingNetwork(state_size, action_size, seed).to(device)
            self.target_network = DuelingNetwork(state_size, action_size, seed).to(device)
        else:
            self.q_network = Network(state_size, action_size, seed).to(device)
            self.target_network = Network(state_size, action_size, seed).to(device)
        self.target_update()

        self.replay_memory = ReplayBuffer(buffer_size=2**17, seed=seed, alpha=pr_alpha, initial_beta=pr_initial_beta,
                                          n_step=multi_step, gamma=gamma)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=5e-4)

        self.epsilon = 1

        self.pr_alpha = pr_alpha
        self.pr_initial_beta = pr_initial_beta
        self.pr_max_beta = pr_max_beta

        self.multi_step = multi_step

    def step(self, state, action, reward, next_state, done):
        self.replay_memory.add(state, action, reward, next_state, done)
        if len(self.replay_memory) >= self.batch_size:
            self.learn()

    def act(self, state):
        if np.random.rand() > self.epsilon:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.q_network.eval()
            with torch.no_grad():
                action_values = self.q_network(state)
            self.q_network.train()
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = np.random.randint(self.action_size)
        return action

    def end_episode(self):
        self.epsilon = max(self.epsilon * 0.995, 0.01)
        self.replay_memory.beta = min(self.pr_max_beta, self.replay_memory.beta + 0.001)

    def learn(self):
        batch, indexes, weights = self.replay_memory.sample(self.batch_size)
        states, actions, rewards, n_states, dones = (torch.from_numpy(v) for v in batch)
        states = states.float().to(device)
        actions = actions.to(device)
        rewards = rewards.float().to(device)
        n_states = n_states.float().to(device)
        dones = dones.float().to(device)

        weights = torch.from_numpy(weights).float().to(device)

        if self.double:
            argmax = self.q_network(n_states).detach().argmax(1).unsqueeze(1)
            max_next_state_q = self.target_network(n_states).detach().gather(1, argmax)
        else:
            max_next_state_q = self.target_network(n_states).detach().max(1)[0].unsqueeze(1)

        target = rewards + (self.gamma**self.multi_step * max_next_state_q * (1 - dones))
        q = self.q_network(states).gather(1, actions)
        loss = torch.abs_(q - target)
        new_priorities = loss.detach().cpu().numpy() + 1e-3
        self.replay_memory.update_priorities(indexes, new_priorities)
        loss = (loss**2 * weights).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.is_soft_target_update:
            self.soft_target_update()
        elif self.num_learn % self.target_update_every:
            self.target_update()

        self.num_learn += 1

    def target_update(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def soft_target_update(self):
        for target_param, q_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.soft_update_ratio * q_param + (1 - self.soft_update_ratio) * target_param)

    def save_network(self, save_dir):
        torch.save(self.q_network.state_dict(), f'{save_dir}/network.pth')

    def load_network(self, filename):
        self.q_network.load_state_dict(torch.load(filename))
        self.target_update()