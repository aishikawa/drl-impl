import random
from collections import deque, namedtuple
from typing import Tuple
import numpy as np

Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    def __init__(self, buffer_size: int, seed: int, alpha: float, initial_beta: float, n_step: int, gamma: float):
        self.memory = SumTree(buffer_size)
        random.seed(seed)

        self.alpha = alpha
        self.beta = initial_beta

        self.n_step = n_step
        self.transitions = deque(maxlen=n_step)

        self.gamma = gamma

    def add(self, state, action, reward, next_state, done) -> None:
        e = Experience(state, action, reward, next_state, int(done))
        self.transitions.append(e)

        if done:
            while self.transitions:
                self.memory.add(self.make_n_step_transition(), self.memory.max_priority)
                self.transitions.popleft()
        elif len(self.transitions) == self.n_step:
            self.memory.add(self.make_n_step_transition(), self.memory.max_priority)

    def make_n_step_transition(self):
        state = self.transitions[0][0]
        action = self.transitions[0][1]
        reward = 0
        for i, t in enumerate(self.transitions):
            reward += t[2] * self.gamma**i
        next_state = self.transitions[-1][3]
        done = self.transitions[-1][4]

        return state, action, reward, next_state, done

    def sample(self, batch_size: int) -> Tuple:
        total_priority = self.memory.total_priority()
        rands = np.random.rand(batch_size) * total_priority
        experiences = [self.memory.get(r) for r in rands]

        indexes, priorities, experiences = zip(*experiences)
        batch = Experience(*zip(*experiences))

        states = np.vstack(batch.state)
        actions = np.vstack(batch.action)
        rewards = np.vstack(batch.reward)
        next_states = np.vstack(batch.next_state)
        dones = np.vstack(batch.done)

        probabilities = priorities / self.memory.total_priority()
        weights = np.power(len(self.memory) * probabilities, -self.beta)
        weights = weights / weights.max()

        return (states, actions, rewards, next_states, dones), indexes, weights

    def update_priorities(self, indexes, new_priorities):
        for i, p in zip(indexes, new_priorities**self.alpha):
            self.memory.update_priority(i, p)

    def __len__(self):
        return len(self.memory)


class SumTree:
    def __init__(self, capacity):
        self.write_index = 0
        self.size = 0
        self.capacity = capacity
        self.tree = np.zeros(2*capacity-1)
        self.data = np.zeros(capacity, dtype=object)
        self.max_priority = 1

    def _update(self, index):
        parent = (index - 1) // 2
        left = 2*parent+1
        right = left+1
        self.tree[parent] = self.tree[left] + self.tree[right]
        if parent != 0:
            self._update(parent)

    def _get(self, index, r):
        left = 2*index+1
        if left >= len(self.tree):
            return index
        if r <= self.tree[left]:
            return self._get(left, r)
        else:
            right = left+1
            return self._get(right, r - self.tree[left])

    def total_priority(self):
        return self.tree[0]

    def add(self, data, priority):
        index = self.write_index + self.capacity - 1
        self.data[self.write_index] = data
        self.update_priority(index, priority)
        self.write_index = (self.write_index + 1) % self.capacity
        self.size += 1
        self.size = min(self.size, self.capacity)

    def update_priority(self, index, priority):
        self.max_priority = max(priority, self.max_priority)
        self.tree[index] = priority
        self._update(index)

    def get(self, r):
        """
        Return data with a probability proportional to the priority
        :param r: random number
        :return: (index, priority, data)
        """
        index = self._get(0, r)
        return index, self.tree[index], self.data[index - self.capacity + 1]

    def __len__(self):
        return self.size
