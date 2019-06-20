
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
    if isinstance(observation_shape, gym.spaces.Box):
      input_shape = observation_shape.shape

    if isinstance(action_shape, gym.spaces.Discrete):
      num_output_neurons = action_shape.n
      self._discrete = True
    self._configure(input_shape, num_output_neurons)

  def forward_pass(self, inputs):
    activations = self._forward_pass(np.array([inputs]))
    if self._discrete:
      return tf.random.categorical(activations, num_samples=1).numpy().flatten()[0]
    else:
      return activations
    

class FeedForwardNetwork(Network):
  def __init__(self, hidden_layers):
    super(FeedForwardNetwork, self).__init__()
    self.hidden_layers = hidden_layers
    
  def _configure(self, input_shape, num_output_neurons, output_activation = tf.nn.softmax):
    self.input_layer = layers.Input(input_shape)

    last_layer = tf.keras.layers.Flatten()(self.input_layer)
    for num_neurons in self.hidden_layers:
      last_layer = layers.Dense(num_neurons, activation=tf.nn.relu)(last_layer)

    last_layer = layers.Dense(num_output_neurons, activation=output_activation)(last_layer)

    self.model = tf.keras.Model(inputs=self.input_layer, outputs=last_layer)

  def _forward_pass(self, inputs):
    return self.model(inputs)

  def summary():
    model.summary()

