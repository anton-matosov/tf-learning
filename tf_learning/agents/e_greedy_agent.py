import tensorflow as tf
import numpy as np

from tf_learning.common import *
from tf_learning.agents import AbstractAgent


class EpsilonAgent(RandomAgent):
  def __init__(self,
    env,
    greedyAgent,
    teacher,
    epsilon_max = .9,
    epsilon_min = .1,
    epsilon_last_step = 3 * 1000):
    super().__init__(env)

    self.teacher = teacher
    self.greedyAgent = greedyAgent
    self.epsilon_max = epsilon_max
    self.epsilon_min = epsilon_min
    self.epsilon_last_step = epsilon_last_step
    self.epsilon_space = np.linspace(self.epsilon_max, self.epsilon_min, self.epsilon_last_step)

  @property
  def epsilon(self):
    current_step = self.teacher.train_step_counter.numpy()
    return self.epsilon_space[np.minimum(current_step, self.epsilon_last_step - 1)]

  def select_action(self, observation):
    explore = self.teacher.force_explore() or np.random.random() < self.epsilon
    with tf.name_scope('Agent/'):
      tf.compat.v2.summary.scalar(name='epsilon', data=self.epsilon, step=self.teacher.train_step_counter)
      tf.compat.v2.summary.scalar(name='explore', data=explore, step=self.teacher.train_step_counter)

    if explore:
      return super().select_action(observation)
    else:
      return self.greedyAgent.select_action(observation)
