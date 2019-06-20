from . abstract_agent import AbstractAgent

class NetworkAgent(AbstractAgent):
  def __init__(self, env, network):
    self.network = network
    return super().__init__(env)

  def select_action(self, observation):
    return self.network.forward_pass(observation)

