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

import time

gpus = tf.config.list_physical_devices('GPU')
print("GPUs:", gpus)
try:
  tf.config.experimental.set_memory_growth(gpus[0], True) # Use only as much GPU memory as needed
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

summaries_dir = "runs"

date_mark = datetime.now().strftime("%Y-%m-%d--%H-%M-%S.%f")
run_dir = path.join(summaries_dir, 'dqn', date_mark)
writer = tf.compat.v2.summary.create_file_writer(run_dir)
writer.set_as_default()

engine = Engine(
  max_total_steps = 20000,
  max_episodes = 100,
  max_steps_per_episode = 200,
  env_name = 'CartPole-v1',
  # env_name = 'FrozenLake-v0',
)

network = FeedForwardNetwork([100])

dqnAgent = NetworkAgentAllInOne(engine.env, network, DqnTeacher())
network.summary()

# agent = RandomAgent(engine.env)

start = time.perf_counter_ns()
try:
  print("Training started. Press Ctrl+C to interrupt")
  engine.rollout(dqnAgent)
except KeyboardInterrupt:
  print("Training interrupted")

end = time.perf_counter_ns()
print("Training finished in {} seconds".format((end - start) / 1e9))

network.save(path.join(run_dir, 'model.h5'))
