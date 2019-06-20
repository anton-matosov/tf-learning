
class AbstractAgent:
  def __init__(self, env):
    self.env = env
    self.observation_space = env.observation_space
    self.action_space = env.action_space
    # print("observation_space", env.observation_space)
    # print("action_space", env.action_space)

  def select_action(self, observation):
    raise Exception("select_action has to be implemented by derived classes")

  def register_reward(self, observation, reward, done):
    pass

  def episode_started(self):
    pass

  def episode_ended(self):
    pass

