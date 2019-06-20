#!/usr/env/bin python3
import argparse

import gym

from agents import RandomAgent
from engine import Engine


engine = Engine(
  num_episodes = 20,
  num_steps_per_episode = 100,
  env_name = 'CartPole-v0',
)

agent = RandomAgent(engine.env)

engine.rollout(agent)
