import gymnasium as gym

class Engine:
  def __init__(self, env_name, max_total_steps = 20000,
      max_episodes = 3000,
      max_steps_per_episode = 200):
    self.env = gym.make(env_name)
    self.max_total_steps = max_total_steps
    self.max_episodes = max_episodes
    self.max_steps_per_episode = max_steps_per_episode
    self.render = False

  def rollout(self, agent):
    global_step = 0
    for i_episode in range(self.max_episodes):
      self.total_reward = 0
      agent.episode_started()
      observation = self.env.reset()
      for t in range(self.max_steps_per_episode):
          self.render_env()

          observation, done = self.step_env(agent, observation)

          global_step += 1
          if done or global_step > self.max_total_steps:
              break
      agent.episode_ended()

      print("{}. Episode {} finished after {} timesteps. Total reward: {}"
        .format(global_step, i_episode + 1, t + 1, self.total_reward))

      if global_step > self.max_total_steps:
        break

    self.env.close()

  def step_env(self, agent, observation):
    action = agent.select_action(observation)

    # observation - an environment-specific object representing your observation of the environment
    # reward - amount of reward achieved by the previous action
    # done - whether it’s time to reset the environment again
    # info - diagnostic information useful for debugging
    observation, reward, done, info = self.env.step(action)

    agent.register_reward(observation, reward, done)
    self.total_reward += reward

    return observation, done

  def render_env(self):
    if self.render:
      env.render()
