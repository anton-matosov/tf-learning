from . abstract_agent import AbstractAgent

import tensorflow as tf

# Experience = collections.namedtuple('Experience', field_names=['observation', 'action', 'reward', 'done', 'new_state'])

class NetworkAgent(AbstractAgent):
  def __init__(self, env, network):
    super().__init__(env)

    self.network = network
    self.network.configure(self.observation_space, self.action_space)

  def select_action(self, observation):
    q_values = self.network.forward_pass(observation)
    return tf.math.argmax(q_values[0]).numpy()

  def register_reward(self, observation, reward, done):
    pass

