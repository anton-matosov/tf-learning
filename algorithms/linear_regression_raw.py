#!/usr/bin/env python3
#%%
import matplotlib.pyplot as plt

from datetime import datetime

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

def loss_tf(predicted_y, expected_y):
  return tf.losses.mean_squared_error(predicted_y, expected_y)

# Some synthetic data
TRUE_W = 3.
TRUE_B = 2.

DATASET_SIZE = 1000
TEST_DATASET_SIZE = 100

inputs = tf.random.normal((DATASET_SIZE,))
noise = tf.random.normal((DATASET_SIZE,))

test_inputs = tf.random.normal((TEST_DATASET_SIZE,))
test_noise = tf.random.normal((TEST_DATASET_SIZE,))

outputs = inputs * TRUE_W + TRUE_B + noise
test_outputs = test_inputs * TRUE_W + TRUE_B + test_noise

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
    current_loss = loss_tf(model(inputs), outputs)
  vars = [model.W, model.b]
  grads = tape.gradient(current_loss, vars)
  optimizer.apply_gradients(zip(grads, vars))
  return current_loss
#%%

#%%

model = Model()
historyW, historyB = [], []
historyLoss, testLoss = [], []


summaries_dir = "runs"

date_mark = datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
writer = tf.contrib.summary.create_file_writer(summaries_dir + '/linear_regression/train_using_optimizer_' + date_mark)
writer.set_as_default()

NUM_EPOCHS = 100
LEARNING_RATE = 0.1

optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
  historyW.append(model.W.numpy())
  historyB.append(model.b.numpy())

  # current_loss = train(model, inputs, outputs, LEARNING_RATE)
  current_loss = train_using_optimizer(model, inputs, outputs, optimizer)
  historyLoss.append(current_loss)
  
  testLoss.append(loss_tf(model(test_inputs), test_outputs).numpy())

  with tf.name_scope('Losses/'):
    tf.compat.v2.summary.scalar(name='loss', data=current_loss, step=epoch)

  with tf.name_scope('Model/'):
    tf.compat.v2.summary.histogram(name='W', data=model.W, step=epoch)
    tf.compat.v2.summary.scalar(name='b', data=model.b, step=epoch)


# merged = tf.summary.merge_all()
# train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
# test_writer = tf.summary.FileWriter(summaries_dir + '/test')
# tf.global_variables_initializer().run()

#%%
plt.plot(historyW, 'r',
          historyB, 'b')
plt.plot([TRUE_W] * len(historyW), 'r--')
plt.plot([TRUE_B] * len(historyB), 'b--')

plt.title("Training history")
plt.legend(["W", "B", "true W", "true B"])
plt.show()

plt.plot(historyLoss, 'g')
plt.plot(testLoss, 'y')
plt.legend(["train", "test"])
plt.title("Losses")
plt.show()

plot_data()

#%%


#%%
