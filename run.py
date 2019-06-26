#!/usr/bin/env python3
import argparse

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import gym
from datetime import datetime
from os import path

from agents import RandomAgent, NetworkAgent
from engine import Engine
from networks import FeedForwardNetwork

summaries_dir = "runs"

date_mark = datetime.now().strftime("%Y-%m-%d--%H-%M-%S.%f")
writer = tf.contrib.summary.create_file_writer(path.join(summaries_dir, 'dqn', date_mark))
writer.set_as_default()

engine = Engine(
  num_episodes = 3000,
  num_steps_per_episode = 200,
  env_name = 'CartPole-v0',
)

network = FeedForwardNetwork([100])

agent = NetworkAgent(engine.env, network)
network.summary()

# agent = RandomAgent(engine.env)


engine.rollout(agent)
