#!/usr/bin/env python3
#%%
import matplotlib.pyplot as plt

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

class Model:
  def __init__(self):
    self.W = tf.Variable(5.)
    self.b = tf.Variable(0.)

  def __call__(self, input):
    return input * self.W + self.b

model = Model()

assert model(3.0).numpy() == 15.0

def loss(predicted_y, expected_y):
  return tf.reduce_mean(tf.square(predicted_y - expected_y))

# Some synthetic data
TRUE_W = 3.
TRUE_B = 2.

DATASET_SIZE = 1000

inputs = tf.random.normal((DATASET_SIZE,))
noise = tf.random.normal((DATASET_SIZE,))

outputs = inputs * TRUE_W + TRUE_B + noise

#%%
def plot_data():
  plt.scatter(inputs, outputs, c='b')
  plt.scatter(inputs, model(inputs), c='r')
  
  plt.show()

plot_data()

#%%
print("Current loss:")
print(loss(model(inputs), outputs))

#%%
def train(model, input, output, learning_rate):
  with tf.GradientTape() as tape:
    current_loss = loss(model(inputs), outputs)
  dW, dB = tape.gradient(current_loss, [model.W, model.b])
  model.W.assign_sub(dW * learning_rate)
  model.b.assign_sub(dB * learning_rate)

  return current_loss
# 

#%%
def train_using_optimizer(model, inputs, outputs, optimizer):
  with tf.GradientTape() as tape:
    current_loss = loss(model(inputs), outputs)
  vars = [model.W, model.b]
  grads = tape.gradient(current_loss, vars)
  optimizer.apply_gradients(zip(grads, vars))
  return current_loss
#%%

model = Model()
historyW, historyB, historyLoss = [], [], []

#%%


NUM_EPOCHS = 1000
LEARNING_RATE = 0.1

optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
  historyW.append(model.W.numpy())
  historyB.append(model.b.numpy())

  # current_loss = train(model, inputs, outputs, LEARNING_RATE)
  current_loss = train_using_optimizer(model, inputs, outputs, optimizer)
  historyLoss.append(current_loss)
  
  # print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' % (epoch, model.W.numpy(), model.b.numpy(), current_loss))

#%%
plt.plot(historyW, 'r',
          historyB, 'b')
plt.plot([TRUE_W] * len(historyW), 'r--')
plt.plot([TRUE_B] * len(historyB), 'b--')

plt.title("Training history")
plt.legend(["W", "B", "true W", "true B"])
plt.show()

plt.plot(historyLoss, 'g')
plt.title("Loss")
plt.show()

plot_data()

#%%
