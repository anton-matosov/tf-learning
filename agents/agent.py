

class RandomAgent:
  def __init__(self, env):
    self.env = env

  def select_action(self, observation):
    return self.env.action_space.sample()

  def register_reward(self, observation, reward, done):
    pass
