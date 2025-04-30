import gymnasium as gym
import numpy as np


class ActionRepeatWrapper(gym.Wrapper):
    """
    Repeat the same action for multiple steps
    """

    def __init__(self, env, repeat=8):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0
        for i in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        return obs, total_reward, terminated, truncated, info


class AverageRewardEarlyStopWrapper(gym.Wrapper):
    """
    End episode early if average reward over last some steps is too low
    """

    def __init__(self, env):
        super().__init__(env)
        self.window_size = 50
        self.threshold = -0.1
        self.recent_rewards = []

    def reset(self, **kwargs):
        self.recent_rewards = []
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.recent_rewards.append(reward)
        self.recent_rewards = self.recent_rewards[-self.window_size :]

        if len(self.recent_rewards) >= self.window_size:
            avg_reward = np.mean(self.recent_rewards)
            if avg_reward <= self.threshold:
                terminated = True

        return obs, reward, terminated, truncated, info


class DieStateRewardWrapper(gym.Wrapper):
    """
    Don't penalize "die state" and add bonus reward if terminated
    """

    def __init__(self, env, bonus_reward=100):
        super().__init__(env)
        self.bonus_reward = bonus_reward

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            reward += self.bonus_reward

        return obs, reward, terminated, truncated, info
