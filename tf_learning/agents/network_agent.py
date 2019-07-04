import tensorflow as tf
import numpy as np

from tf_learning.common import *
from tf_learning.agents import AbstractAgent


EPSILON_MAX = .9
EPSILON_MIN = .1
EPSILON_LAST_STEP = 3 * 1000
EPSILON_SPACE = np.linspace(EPSILON_MAX, EPSILON_MIN, EPSILON_LAST_STEP)

class NetworkAgent(AbstractAgent):
  def __init__(self, env, network, teacher):
    super().__init__(env)

    self.teacher = teacher

    self.network = network
    self.network.configure(self.observation_space, self.action_space, tf.nn.relu)

  @property
  def epsilon(self):
    current_step = self.teacher.train_step_counter.numpy()
    return EPSILON_SPACE[np.minimum(current_step, EPSILON_LAST_STEP - 1)]

  def select_action(self, observation):
    self.last_observation = observation
    with tf.name_scope('Agent/'):
      tf.compat.v2.summary.scalar(name='epsilon', data=self.epsilon, step=self.teacher.train_step_counter)

    if self.teacher.force_explore() or np.random.random() < self.epsilon:
      self.last_action = self.env.action_space.sample()
      with tf.name_scope('Agent/'):
        tf.compat.v2.summary.scalar(name='explore', data=True, step=self.teacher.train_step_counter)
    else:
      with tf.name_scope('Agent/'):
        tf.compat.v2.summary.scalar(name='explore', data=False, step=self.teacher.train_step_counter)
      q_values = self.network.forward_pass(np.expand_dims(observation, 0))[0]
      self.last_action = tf.math.argmax(q_values).numpy()

    return self.last_action

  def register_reward(self, observation, reward, done):
    experience = Experience(self.last_observation, self.last_action, reward, done, observation)
    self.teacher.record_experience(experience)

    self.teacher.learn(self.network)

  # def episode_ended(self):
    # print("Trained steps: ", self.teacher.train_step_counter.numpy())
  #   self.teacher.learn(self.network)
