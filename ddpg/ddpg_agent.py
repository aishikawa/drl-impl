from ddpg.replay_buffer import ReplayBuffer
from ddpg.network import ActorNetwork, CriticNetwork
import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DdpgAgent:
    def __init__(self, state_size, action_size, gamma, random_seed, buffer_size=100000, batch_size=128,
                 soft_update_ratio=0.001, lr_actor=3e-4, lr_critic=3e-4, weight_decay=0):
        self.gamma = gamma
        self.soft_update_ratio = soft_update_ratio

        self.actor = ActorNetwork(state_size, action_size, random_seed).to(device)
        self.actor_target = ActorNetwork(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = CriticNetwork(state_size, action_size, random_seed).to(device)
        self.critic_target = CriticNetwork(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=weight_decay)

        self.noise = OUNoise(action_size)

        self.memory = ReplayBuffer(buffer_size)

        self.batch_size = batch_size

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) >= self.batch_size:
            self.learn()

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state)
        self.actor.train()
        action = action.cpu().data.numpy()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def end_episode(self):
        self.noise.reset()

    def learn(self):
        experiences = self.memory.sample(self.batch_size)
        states, actions, rewards, n_states, dones = (torch.from_numpy(v) for v in experiences)
        states = states.float().to(device)
        actions = actions.float().to(device)
        rewards = rewards.float().to(device)
        next_states = n_states.float().to(device)
        dones = dones.float().to(device)

        # update critic
        next_actions = self.actor_target(next_states)
        next_q = self.critic_target(next_states, next_actions)
        target_q = rewards + (self.gamma * next_q * (1-dones))
        q = self.critic(states, actions)
        critic_loss = F.mse_loss(q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update target
        for t_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            t_param.data.copy_(self.soft_update_ratio * param.data + (1 - self.soft_update_ratio) * t_param.data)
        for t_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            t_param.data.copy_(self.soft_update_ratio * param.data + (1 - self.soft_update_ratio) * t_param.data)

    def save_network(self, save_dir):
        torch.save(self.actor.state_dict(), f'{save_dir}/actor_network.pth')
        torch.save(self.critic.state_dict(), f'{save_dir}/critic_network.pth')


class OUNoise:
    """
    Ornstein-Uhlenbeck process.
    """
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma

        self.state = copy.copy(self.mu)

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for _ in range(self.size)])
        self.state = x + dx
        return self.state
