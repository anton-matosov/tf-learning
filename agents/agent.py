
class AbstractAgent:
  def __init__(self, env):
    self.env = env

  def select_action(self, observation):
    raise Exception("select_action has to be implemented by derived classes")

  def register_reward(self, observation, reward, done):
    pass

  def episode_started(self):
    pass

  def episode_ended(self):
    pass


class RandomAgent(AbstractAgent):
  def select_action(self, observation):
    return self.env.action_space.sample()

