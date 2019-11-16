import random
from collections import deque
from typing import List, Tuple


class ReplayBuffer:
    def __init__(self, buffer_size: int, seed: int):
        self.memory = deque(maxlen=buffer_size)
        random.seed(seed)

    def add(self, state, action, reward, next_state, done) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
