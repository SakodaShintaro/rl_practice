import cv2
import gymnasium as gym
import minigrid
import numpy as np

REPEAT = 8


def make_env(env_id: str, partial_obs: bool) -> gym.Env:
    env = gym.make(env_id, render_mode="rgb_array")

    if env_id.startswith("MiniGrid"):
        if partial_obs:
            env = minigrid.wrappers.RGBImgPartialObsWrapper(env)
        else:
            env = minigrid.wrappers.RGBImgObsWrapper(env)
        env = minigrid.wrappers.ImgObsWrapper(env)
        env = DiscreteToContinuousWrapper(env)
        env = ResizeObs(env, shape=(3, 96, 96))
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = TransposeAndNormalizeObs(env)
        return env

    elif env_id.startswith("CarRacing"):
        env = env.env  # Unwrap the original TimeLimit wrapper
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000 * REPEAT)
        env = CarRacingRewardFixWrapper(env)
        env = ActionRepeatWrapper(env, repeat=REPEAT)
        env = AverageRewardEarlyStopWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = TransposeAndNormalizeObs(env)
        return env

    else:
        raise ValueError(f"Unsupported environment: {env_id}")


class DiscreteToContinuousWrapper(gym.Wrapper):
    """
    Convert discrete action space to continuous.
    Example: Discrete(4) -> Box(-1.0, 1.0, (4,))
    The max action is selected.
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(env.action_space.n,), dtype=np.float32
        )

    def step(self, action):
        # Convert continuous action to discrete by selecting the max action
        discrete_action = np.argmax(action)
        return self.env.step(discrete_action)


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
        o = obs.astype(np.float32) / 255.0
        # Convert from (H, W, C) to (C, H, W)
        o = np.transpose(o, (2, 0, 1))
        return o


class CarRacingRewardFixWrapper(gym.Wrapper):
    """
    Fix CarRacing's -100 penalty for going off-track.
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Fix the -100 penalty
        if reward < -30:
            reward += 100

        return obs, reward, terminated, truncated, info


class ResizeObs(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = shape
        h, w = shape[1:]  # shape is (C, H, W), so extract H, W
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(h, w, 3), dtype=np.float32
        )

    def observation(self, obs):
        # obs is (H, W, C), resize and return (H, W, C)
        h, w = self.shape[1:]  # target height and width
        return cv2.resize(obs, (w, h), interpolation=cv2.INTER_AREA)
