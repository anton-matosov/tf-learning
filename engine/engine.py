import gym

class Engine:
  def __init__(self, env_name, num_episodes = 20, num_steps_per_episode = 100):
    self.env = gym.make(env_name)
    self.num_episodes = num_episodes
    self.num_steps_per_episode = num_steps_per_episode
    self.render = False
    
  def rollout(self, agent):
    for i_episode in range(self.num_episodes):
      agent.episode_started()
      observation = self.env.reset()
      for t in range(self.num_steps_per_episode):
          self.render_env()

          observation, done = self.step_env(agent, observation)

          if done:
              print("Episode finished after {} timesteps".format(t+1))
              break
      agent.episode_ended()

    self.env.close()

  def step_env(self, agent, observation):
    action = agent.select_action(observation)

    # observation - an environment-specific object representing your observation of the environment
    # reward - amount of reward achieved by the previous action
    # done - whether it’s time to reset the environment again
    # info - diagnostic information useful for debugging
    observation, reward, done, info = self.env.step(action)

    agent.register_reward(observation, reward, done)
    return observation, done

  def render_env(self):
    if self.render:
      env.render()
