
from . abstract_agent import AbstractAgent


class RandomAgent(AbstractAgent):
  def select_action(self, observation):
    return self.env.action_space.sample()

