

class RandomAgent:
  def __init__(self, env, *args, **kwargs):
    self.env = env
    return super().__init__(*args, **kwargs)

  def select_action(self, observation):
    return self.env.action_space.sample()

  def register_reward(self, observation, reward, done):
    pass
