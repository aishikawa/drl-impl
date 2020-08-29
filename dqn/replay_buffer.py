import random
from collections import deque, namedtuple
from typing import Tuple
import numpy as np

Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    def __init__(self, buffer_size: int):
        self.memory = deque(maxlen=buffer_size)

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
