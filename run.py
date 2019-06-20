#!/usr/env/bin python3
import argparse

import tensorflow as tf
tf.enable_eager_execution()

import gym

from agents import RandomAgent, NetworkAgent
from engine import Engine
from networks import FeedForwardNetwork


engine = Engine(
  num_episodes = 20,
  num_steps_per_episode = 100,
  env_name = 'CartPole-v0',
)

agent = NetworkAgent(engine.env, FeedForwardNetwork([20, 30]))
# agent = RandomAgent(engine.env)

engine.rollout(agent)
