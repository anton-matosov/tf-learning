from . abstract_agent import AbstractAgent

import tensorflow as tf

import numpy as np
import collections

Experience = collections.namedtuple('Experience', field_names=['observation', 'action', 'reward', 'done', 'new_observation'])

REPLAY_BUFFER_SIZE = 100 * 1000
REPLAY_MIN_SIZE = 1000


LEARNING_RATE = 1e-5
EPSILON_MAX = .9
EPSILON_MIN = .1
EPSILON_LAST_STEP = 15 * 1000
EPSILON_SPACE = np.linspace(EPSILON_MAX, EPSILON_MIN, EPSILON_LAST_STEP)

GAMMA = .9

TARGET_NETWORK_LIFETIME = 10

TRAINING_ITERATIONS = 10
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


def add_variables_summaries(grads_and_vars, step):
  """Add summaries for variables.

  Args:
    grads_and_vars: A list of (gradient, variable) pairs.
    step: Variable to use for summaries.
  """
  with tf.name_scope('summarize_vars'):
    for grad, var in grads_and_vars:
      if grad is not None:
        if isinstance(var, tf.IndexedSlices):
          var_values = var.values
        else:
          var_values = var
        var_name = var.name.replace(':', '_')
        tf.compat.v2.summary.histogram(
            name=var_name + '_value', data=var_values, step=step)
        tf.compat.v2.summary.scalar(
            name=var_name + '_value_norm',
            data=tf.linalg.global_norm([var_values]),
            step=step)


def add_gradients_summaries(grads_and_vars, step):
  """Add summaries to gradients.

  Args:
    grads_and_vars: A list of gradient to variable pairs (tuples).
    step: Variable to use for summaries.
  """
  with tf.name_scope('summarize_grads'):
    for grad, var in grads_and_vars:
      if grad is not None:
        if isinstance(grad, tf.IndexedSlices):
          grad_values = grad.values
        else:
          grad_values = grad
        var_name = var.name.replace(':', '_')
        tf.compat.v2.summary.histogram(
            name=var_name + '_gradient', data=grad_values, step=step)
        tf.compat.v2.summary.scalar(
            name=var_name + '_gradient_norm',
            data=tf.linalg.global_norm([grad_values]),
            step=step)
      # else:
      #   logging.info('Var %s has no gradient', var.name)

def generate_tensor_summaries(tag, tensor, step):
  """Generates various summaries of `tensor` such as histogram, max, min, etc.

  Args:
    tag: A namescope tag for the summaries.
    tensor: The tensor to generate summaries of.
    step: Variable to use for summaries.
  """
  with tf.name_scope(tag):
    tf.compat.v2.summary.histogram(name='histogram', data=tensor, step=step)
    tf.compat.v2.summary.scalar(
        name='mean', data=tf.reduce_mean(input_tensor=tensor), step=step)
    tf.compat.v2.summary.scalar(
        name='mean_abs',
        data=tf.reduce_mean(input_tensor=tf.abs(tensor)),
        step=step)
    tf.compat.v2.summary.scalar(
        name='max', data=tf.reduce_max(input_tensor=tensor), step=step)
    tf.compat.v2.summary.scalar(
        name='min', data=tf.reduce_min(input_tensor=tensor), step=step)


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
      tf.compat.v2.summary.scalar(name='loss', data=loss, step=self.train_step_counter.numpy())

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
      tf.compat.v2.summary.scalar(name='epsilon', data=self.epsilon, step=self.train_step_counter.numpy())

    if self.teacher.force_explore() or np.random.random() < self.epsilon:
      self.last_action = self.env.action_space.sample()
      with tf.name_scope('Agent/'):
        tf.compat.v2.summary.scalar(name='explore', data=True, step=self.train_step_counter.numpy())
    else:
      with tf.name_scope('Agent/'):
        tf.compat.v2.summary.scalar(name='explore', data=False, step=self.train_step_counter.numpy())
      q_values = self.network.forward_pass(np.expand_dims(observation, 0))[0]
      self.last_action = tf.math.argmax(q_values).numpy()

    return self.last_action

  def register_reward(self, observation, reward, done):
    experience = Experience(self.last_observation, self.last_action, reward, done, observation)
    self.teacher.record_experience(experience)

    self.teacher.learn(self.network)

  def episode_ended(self):
    print("Trained steps: ", self.teacher.train_step_counter.numpy())
  #   self.teacher.learn(self.network)
