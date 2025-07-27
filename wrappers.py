import gymnasium as gym
import numpy as np

REPEAT = 8


def make_env():
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    env = env.env  # Unwrap the original TimeLimit wrapper
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000 * REPEAT)
    env = ActionRepeatWrapper(env, repeat=REPEAT)
    env = AverageRewardEarlyStopWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = TransposeAndNormalizeObs(env)
    return env


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
        self.window_size = 25
        self.recent_rewards = []

    def reset(self, **kwargs):
        self.recent_rewards = []
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.recent_rewards.append(reward)
        self.recent_rewards = self.recent_rewards[-self.window_size :]

        if len(self.recent_rewards) >= self.window_size:
            count = sum(r < 0.0 for r in self.recent_rewards)
            if count == self.window_size:
                truncated = True

        return obs, reward, terminated, truncated, info


class TransposeAndNormalizeObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        h, w = env.observation_space.shape[0:2]
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(3, h, w), dtype=np.float32
        )

    def observation(self, obs):
        # obs: (H, W, 3)
        o = obs.astype(np.float32) / 255.0
        o = np.transpose(o, (2, 0, 1))  # (3, H, W)
        return o
