#!/usr/bin/env python3
import argparse

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import gym

from agents import RandomAgent, NetworkAgent
from engine import Engine
from networks import FeedForwardNetwork


engine = Engine(
  num_episodes = 2000,
  num_steps_per_episode = 200,
  env_name = 'CartPole-v0',
)

network = FeedForwardNetwork([100])

agent = NetworkAgent(engine.env, network)
network.summary()

# agent = RandomAgent(engine.env)


engine.rollout(agent)
