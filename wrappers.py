import gymnasium as gym
import numpy as np

STACK_SIZE = 1
REPEAT = 8


def make_env(video_dir):
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    env = env.env  # Unwrap the original TimeLimit wrapper
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000 * REPEAT)
    env = gym.wrappers.FrameStackObservation(env, stack_size=STACK_SIZE)
    env = ActionRepeatWrapper(env, repeat=REPEAT)
    env = AverageRewardEarlyStopWrapper(env)
    env = DieStateRewardWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.RecordVideo(
        env, video_folder=video_dir, episode_trigger=lambda x: x % 100 == 0
    )
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


class TransposeAndNormalizeObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        h, w = env.observation_space.shape[1:3]
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(STACK_SIZE * 3, h, w), dtype=np.float32
        )

    def observation(self, obs):
        # obs: (STACK_SIZE, H, W, 3)
        o = obs.astype(np.float32) / 255.0
        o = np.transpose(o, (0, 3, 1, 2))  # (STACK_SIZE, 3, H, W)
        return o.reshape(-1, o.shape[2], o.shape[3])
