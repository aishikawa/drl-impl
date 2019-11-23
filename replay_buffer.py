import random
from collections import deque
from typing import List, Tuple
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size: int, seed: int):
        self.memory = deque(maxlen=buffer_size)
        random.seed(seed)

    def add(self, state, action, reward, next_state, done) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        experiences = random.sample(self.memory, batch_size)
        experiences = np.array(experiences)
        return experiences[: 0], experiences[:, 1], experiences[:, 2], experiences[:, 3], experiences[:, 4]

    def __len__(self):
        return len(self.memory)
