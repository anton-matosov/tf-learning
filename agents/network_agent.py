from . abstract_agent import AbstractAgent

import tensorflow as tf

import numpy as np
import collections

Experience = collections.namedtuple('Experience', field_names=['observation', 'action', 'reward', 'done', 'new_observation'])

REPLAY_BUFFER_SIZE = 10 * 1000
REPLAY_MIN_SIZE = 20
# REPLAY_MIN_SIZE = REPLAY_BUFFER_SIZE / 10

TARGET_NETWORK_LIFETIME = 100

GAMMA = 0.9
BATCH_SIZE = 10

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
               np.array(dones, dtype=np.uint8), np.array(next_observations)


class DqnTeacher:
  def __init__(self):
    self.buffer = ExperienceBuffer(REPLAY_BUFFER_SIZE)
    self._target_network_used = TARGET_NETWORK_LIFETIME

  def record_experience(self, experience):
    self.buffer.append(experience)

  def learn(self, qnetwork):
    if len(self.buffer) < REPLAY_MIN_SIZE:
      return

    self._target_network_used += 1
    if self._target_network_used >= TARGET_NETWORK_LIFETIME:
      self._target_network_used = 0
      self._target_network = qnetwork.clone()

    batch = self.buffer.sample(BATCH_SIZE)
    loss_t = self.calc_loss(batch, qnetwork, self._target_network)
    # loss_t.backward()
    # optimizer.step()

  def calc_loss(self, batch, net, tgt_net):
    states, actions, rewards, dones, next_states = batch

    q_values = net.forward_pass(states)
    actions_tf = tf.expand_dims(actions, -1)
    state_action_values = tf.gather(q_values, actions_tf, batch_dims=1)
    state_action_values_flat = tf.squeeze(state_action_values, -1)
    # print("q_values", q_values, "actions_tf", actions_tf)
    # print("state_action_values", state_action_values)

    next_q_values = tgt_net.forward_pass(next_states)
    next_state_values = tf.math.reduce_max(next_q_values, axis=1)
    
    expected_state_action_values = next_state_values * GAMMA + rewards

    losses = tf.squared_difference(state_action_values_flat, expected_state_action_values)
    loss = tf.reduce_mean(losses)
    return loss

class NetworkAgent(AbstractAgent):
  def __init__(self, env, network):
    super().__init__(env)

    self.teacher = DqnTeacher()

    self.network = network
    self.network.configure(self.observation_space, self.action_space)

  def select_action(self, observation):
    q_values = self.network.forward_pass(np.array([observation]))
    self.last_observation = observation
    self.last_action = tf.math.argmax(q_values[0]).numpy()
    return self.last_action

  def register_reward(self, observation, reward, done):
    experience = Experience(self.last_observation, self.last_action, reward, done, observation)
    self.teacher.record_experience(experience)

  def episode_ended(self):
    self.teacher.learn(self.network)
