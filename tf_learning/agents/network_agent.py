import tensorflow as tf
import numpy as np

from tf_learning.common import *
from tf_learning.agents import AbstractAgent


class NetworkAgentAllInOne(AbstractAgent):
  def __init__(self, env, network, teacher):
    super().__init__(env)

    self.epsilon_max = .9
    self.epsilon_min = .02
    self.epsilon_last_step = 3 * 1000
    self.epsilon_space = np.linspace(self.epsilon_max, self.epsilon_min, self.epsilon_last_step)

    self.teacher = teacher

    self.network = network
    self.network.configure(self.observation_space, self.action_space, tf.nn.relu)

  @property
  def epsilon(self):
    current_step = self.teacher.train_step_counter.numpy()
    return self.epsilon_space[np.minimum(current_step, self.epsilon_last_step - 1)]

  def select_action(self, observation):
    self.last_observation = observation

    explore = self.teacher.force_explore() or np.random.random() < self.epsilon
    with tf.name_scope('Agent/'):
      tf.compat.v2.summary.scalar(name='epsilon', data=self.epsilon, step=self.teacher.train_step_counter)
      tf.compat.v2.summary.scalar(name='explore', data=explore, step=self.teacher.train_step_counter)

    if explore:
      self.last_action = self.env.action_space.sample()
    else:
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
