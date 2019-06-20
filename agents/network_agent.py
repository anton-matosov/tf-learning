from . abstract_agent import AbstractAgent

import tensorflow as tf

class NetworkAgent(AbstractAgent):
  def __init__(self, env, network):
    super().__init__(env)

    self.network = network
    self.network.configure(self.observation_space, self.action_space)
    

  def select_action(self, observation):
    probs = self.network.forward_pass(observation)
    return tf.random.categorical(probs, 1).numpy()[0][0]

