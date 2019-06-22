from . abstract_agent import AbstractAgent

import tensorflow as tf

import numpy as np
import collections

Experience = collections.namedtuple('Experience', field_names=['observation', 'action', 'reward', 'done', 'new_observation'])

REPLAY_BUFFER_SIZE = 100 * 1000
REPLAY_MIN_SIZE = 1000


LEARNING_RATE = 1e-8
EPSILON = .1

GAMMA = 1.

TARGET_NETWORK_LIFETIME = 10

TRAINING_ITERATIONS = 5
BATCH_SIZE = 32

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
    self.optimizer = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE)
    # self.optimizer = tf.keras.optimizers.RMSprop(LEARNING_RATE)
    self.train_step_counter = tf.Variable(0, trainable=False, name='train_step_counter')

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

      # print("loss", loss)
      self.optimizer.apply_gradients(
        zip(gradients, qnetwork.model.trainable_variables),
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
    
    td_error = valid_mask * (expected_state_action_values - state_action_values_flat)
    
    # losses = tf.math.squared_difference(state_action_values_flat, expected_state_action_values)
    losses = valid_mask * tf.compat.v1.losses.mean_squared_error(state_action_values_flat, expected_state_action_values)
    loss = tf.reduce_mean(losses)
    return loss

class NetworkAgent(AbstractAgent):
  def __init__(self, env, network):
    super().__init__(env)

    self.teacher = DqnTeacher()

    self.epsilon = EPSILON
    self.network = network
    self.network.configure(self.observation_space, self.action_space, tf.nn.relu)

  def select_action(self, observation):
    self.last_observation = observation
    if self.teacher.force_explore() or np.random.random() < self.epsilon:
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
  #   self.teacher.learn(self.network)
