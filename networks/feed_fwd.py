
import tensorflow as tf
from tensorflow.keras import layers
import gym
import numpy as np

class Network():
  def forward_pass(self, input):
    pass

  def learn(self):
    pass

  def configure(self, observation_shape, action_shape, output_activation = tf.nn.softmax):
    if isinstance(action_shape, gym.spaces.Discrete):
      num_output_neurons = action_shape.n
    if isinstance(observation_shape, gym.spaces.Box):
      num_input_neurons = 1
      for x in observation_shape.shape:
        num_input_neurons *= x
    self._configure(num_input_neurons, num_output_neurons)


class FeedForwardNetwork(Network):
  def __init__(self, hidden_layers):
    super(FeedForwardNetwork, self).__init__()
    self.hidden_layers = hidden_layers
    
  def _configure(self, num_input_neurons, num_output_neurons, output_activation = tf.nn.softmax):
    self.input_layer = layers.Input(shape=(num_input_neurons,))

  # tf.keras.layers.Flatten(),
    last_layer = self.input_layer
    for num_neurons in self.hidden_layers:
      last_layer = layers.Dense(num_neurons, activation=tf.nn.relu)(last_layer)

    last_layer = layers.Dense(num_output_neurons, activation=output_activation)(last_layer)

    self.model = tf.keras.Model(inputs=self.input_layer, outputs=last_layer)

  def forward_pass(self, inputs):
    return self.model(np.array([inputs]))

  def summary():
    model.summary()

