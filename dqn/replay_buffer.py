import random
from collections import deque, namedtuple
from typing import Tuple
import numpy as np

Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    def __init__(self, buffer_size: int, seed: int):
        self.memory = deque(maxlen=buffer_size)
        random.seed(seed)

    def add(self, state, action, reward, next_state, done) -> None:
        e = Experience(state, action, reward, next_state, int(done))
        self.memory.append(e)

    def sample(self, batch_size: int) -> Tuple:
        experiences = random.sample(self.memory, batch_size)
        batch = Experience(*zip(*experiences))
        states = np.vstack(batch.state)
        actions = np.vstack(batch.action)
        rewards = np.vstack(batch.reward)
        next_states = np.vstack(batch.next_state)
        dones = np.vstack(batch.done)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class SumTree:
    def __init__(self, capacity):
        self.write_index = 0
        self.size = 0
        self.capacity = capacity
        self.tree = np.zeros(2*capacity-1)
        self.data = np.zeros(capacity, dtype=object)
        self.max_priority = 0

    def _update(self, index, change):
        parent = (index - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._update(parent, change)

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
        change = priority - self.tree[index]
        self.tree[index] = priority
        self._update(index, change)

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
