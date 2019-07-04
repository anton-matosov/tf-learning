import numpy as np
import collections

Experience = collections.namedtuple('Experience', field_names=['observation', 'action', 'reward', 'done', 'new_observation'])

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        observations, actions, rewards, dones, next_observations = zip(*[self.buffer[idx] for idx in indices])
        return np.array(observations), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.bool), np.array(next_observations)
