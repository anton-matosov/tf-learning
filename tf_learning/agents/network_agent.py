from tf_learning.agents import AbstractAgent

import tensorflow as tf

import numpy as np
import collections

from tf_learning.common.summary import *

Experience = collections.namedtuple('Experience', field_names=['observation', 'action', 'reward', 'done', 'new_observation'])

REPLAY_BUFFER_SIZE = 20 * 1000
REPLAY_MIN_SIZE = 1000


LEARNING_RATE = 1e-6
EPSILON_MAX = .9
EPSILON_MIN = .1
EPSILON_LAST_STEP = 3 * 1000
EPSILON_SPACE = np.linspace(EPSILON_MAX, EPSILON_MIN, EPSILON_LAST_STEP)

GAMMA = .9

TARGET_NETWORK_LIFETIME = 1

TRAINING_ITERATIONS = 1
BATCH_SIZE = 64

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        observations, actions, rewards, dones, next_observations = zip(*[self.buffer[idx] for idx in indices])
        return np.array(observations), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.bool), np.array(next_observations)


class DqnTeacher:
  def __init__(self):
    self.buffer = ExperienceBuffer(REPLAY_BUFFER_SIZE)
    self._target_network = None
    self._target_network_used = TARGET_NETWORK_LIFETIME
    # self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(LEARNING_RATE)
    self.optimizer = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE)
    # self.optimizer = tf.keras.optimizers.RMSprop(LEARNING_RATE)
    self.train_step_counter = tf.Variable(0, trainable=False, name='train_step_counter', dtype=tf.int64)

    self._summarize_grads_and_vars = True
    self._debug_summaries = True

  def record_experience(self, experience):
    self.buffer.append(experience)

  def force_explore(self):
    return len(self.buffer) < REPLAY_MIN_SIZE

  def learn(self, qnetwork):
    if len(self.buffer) < REPLAY_MIN_SIZE:
      return

    if not self._target_network:
      self._target_network = qnetwork.clone()

    for _ in range(TRAINING_ITERATIONS):
      batch = self.buffer.sample(BATCH_SIZE)

      with tf.GradientTape() as tape:
        loss = self.compute_loss(batch, qnetwork, self._target_network)
      gradients = tape.gradient(loss, qnetwork.model.trainable_variables)
      grads_and_vars = tuple(zip(gradients, qnetwork.model.trainable_variables))

      if self._summarize_grads_and_vars:
        add_variables_summaries(grads_and_vars, self.train_step_counter)
        add_gradients_summaries(grads_and_vars, self.train_step_counter)

      self.optimizer.apply_gradients(
        grads_and_vars,
        global_step=self.train_step_counter)

    self._target_network_used += 1
    if self._target_network_used >= TARGET_NETWORK_LIFETIME:
      self._target_network_used = 0
      self._target_network.copy_weights_from(qnetwork)

  def compute_loss(self, batch, net, tgt_net):
    states, actions, rewards, dones, next_states = batch

    q_values = net.forward_pass(states)
    actions_tf = tf.expand_dims(actions, -1)
    state_action_values = tf.gather(q_values, actions_tf, batch_dims=1)
    state_action_values_flat = tf.squeeze(state_action_values, -1)

    next_q_values = tgt_net.forward_pass(next_states)
    next_state_values = tf.math.reduce_max(next_q_values, axis=1)
    
    expected_state_action_values = tf.stop_gradient(next_state_values * GAMMA + rewards)

    valid_mask = tf.cast(~dones, tf.float32)
    
    
    td_loss = valid_mask * tf.compat.v1.losses.mean_squared_error(expected_state_action_values, state_action_values_flat, reduction=tf.compat.v1.losses.Reduction.NONE)
    loss = tf.reduce_mean(td_loss)

    with tf.name_scope('Losses/'):
      tf.compat.v2.summary.scalar(name='loss', data=loss, step=self.train_step_counter)

    if self._debug_summaries:
      td_error = valid_mask * (expected_state_action_values - state_action_values_flat)
      diff_q_values = state_action_values_flat - expected_state_action_values
      generate_tensor_summaries('td_error', td_error,
                                        self.train_step_counter)
      generate_tensor_summaries('td_loss', td_loss,
                                        self.train_step_counter)
      generate_tensor_summaries('q_values', state_action_values_flat,
                                        self.train_step_counter)
      generate_tensor_summaries('next_q_values', expected_state_action_values,
                                        self.train_step_counter)
      generate_tensor_summaries('diff_q_values', diff_q_values,
                                        self.train_step_counter)

    return loss

class NetworkAgent(AbstractAgent):
  def __init__(self, env, network):
    super().__init__(env)

    self.teacher = DqnTeacher()

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
