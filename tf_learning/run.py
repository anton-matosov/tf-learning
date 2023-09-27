#!/usr/bin/env python3
import argparse

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import gymnasium as gym
from datetime import datetime
from os import path

from agents import RandomAgent, NetworkAgentAllInOne
from engine import Engine
from networks import FeedForwardNetwork

from tf_learning.algorithms import DqnTeacher

summaries_dir = "runs"

date_mark = datetime.now().strftime("%Y-%m-%d--%H-%M-%S.%f")
writer = tf.compat.v2.summary.create_file_writer(path.join(summaries_dir, 'dqn', date_mark))
writer.set_as_default()

engine = Engine(
  max_total_steps = 20000,
  max_episodes = 3000,
  max_steps_per_episode = 200,
  env_name = 'CartPole-v0',
  # env_name = 'FrozenLake-v0',
)

network = FeedForwardNetwork([100])

dqnAgent = NetworkAgentAllInOne(engine.env, network, DqnTeacher())
network.summary()

# agent = RandomAgent(engine.env)

engine.rollout(dqnAgent)
