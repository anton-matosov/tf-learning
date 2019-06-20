#!/usr/env/bin python3
import argparse

import gym

from agents import RandomAgent

num_episodes = 20
num_steps_per_episode = 100
env_name = 'CartPole-v0'

env = gym.make(env_name)
agent = RandomAgent(env)

for i_episode in range(num_episodes):
    observation = env.reset()
    for t in range(num_steps_per_episode):
        env.render()
        print(observation)
        action = agent.select_action(observation)

        # observation - an environment-specific object representing your observation of the environment
        # reward - amount of reward achieved by the previous action
        # done - whether it’s time to reset the environment again
        # info - diagnostic information useful for debugging
        observation, reward, done, info = env.step(action)

        agent.register_reward(observation, reward, done)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()