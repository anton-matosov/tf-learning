
import tensorflow as tf
from tensorflow.keras import layers
import gym
import numpy as np

class Network():
  def forward_pass(self, input):
    pass

  def learn(self):
    pass

  def configure(self, observation_shape, action_shape, output_activation = None):
    if isinstance(observation_shape, gym.spaces.Box):
      input_shape = observation_shape.shape
      input_layer = layers.Input(input_shape)

    if isinstance(observation_shape, gym.spaces.Discrete):
      input_shape = observation_shape.n
      input_layer = layers.Embedding(input_dim=input_shape, output_dim=input_shape, input_length=1)

    if isinstance(action_shape, gym.spaces.Discrete):
      num_output_neurons = action_shape.n

    self._configure(input_layer, num_output_neurons, output_activation)

  def forward_pass(self, observations_batch):
    self.activations = self._forward_pass(observations_batch)
    return self.activations




class FeedForwardNetwork(Network):
  def __init__(self, hidden_layers):
    super(FeedForwardNetwork, self).__init__()
    self.hidden_layers = hidden_layers
    
  def _configure(self, input_layer, num_output_neurons, output_activation = None):
    self.model = tf.keras.Sequential()

    self.model.add(input_layer)
    self.model.add(tf.keras.layers.Flatten())

    for num_neurons in self.hidden_layers:
      self.model.add(layers.Dense(num_neurons, activation=tf.nn.relu,
      kernel_initializer=tf.compat.v1.variance_scaling_initializer(
          scale=2.0, mode='fan_in', distribution='truncated_normal')))
    
    # self.model.add(layers.Dense(num_output_neurons, activation=output_activation))
    self.model.add(layers.Dense(num_output_neurons, 
        activation=None,
        kernel_initializer=tf.compat.v1.initializers.random_uniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.compat.v1.initializers.constant(-0.2),
      ))

  def _forward_pass(self, inputs):
    return self.model(inputs)

  def summary(self):
    self.model.summary()

  def clone(self):
    copy = FeedForwardNetwork(self.hidden_layers)
    copy.model = tf.keras.models.clone_model(self.model)
    copy.copy_weights_from(self) 
    return copy

  def copy_weights_from(self, other):
    self.model.set_weights(other.model.get_weights())

