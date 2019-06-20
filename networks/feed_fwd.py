
class Network:
  def forward_pass(self, input):
    pass

  def learn(self):
    pass


import tensorflow as tf
from tf.keras import layers

class FeedForwardNetwork(Network):
  def __init__(self,
    num_input_neurons,
    num_output_neurons,
    hidden_layers = [],
    output_activation = tf.nn.softmax
    ):
    self.input_layer = layers.InputLayer(input_shape=(num_input_neurons,)),

    last_layer = self.input_layer
    for num_neurons in range(hidden_layers):
      last_layer = layers.Dense(units = num_neurons, activation=tf.nn.relu)(last_layer)
    layers.Dense(units = num_output_neurons, activation=output_activation)(last_layer)

  def forward_pass(self, inputs):
    return self.input_layer(inputs)

