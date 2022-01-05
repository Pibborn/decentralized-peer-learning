import numpy as np


class SuggestionBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.size = 0
        self.index = 0
        self.capacity = capacity

    def add(self, reward, suggested_action, followed_agent, next_obs):
        data = [reward, suggested_action, followed_agent, next_obs]
        if (self.size == self.capacity):
            self.buffer[self.index] = data
            self.index = (self.index + 1) % self.capacity
        else:
            self.buffer.append(data)
            self.size += 1

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size)
        return [self.buffer[index] for index in indices]


if __name__ == "__main__":
    buffer = SuggestionBuffer(50)

    for i in range(100):
        buffer.add(i, i % 9, i % 4, {})

    samples = buffer.sample(3)

    print(samples)
