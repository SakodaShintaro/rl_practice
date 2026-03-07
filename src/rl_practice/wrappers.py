# SPDX-License-Identifier: MIT
import re

import cv2
import gymnasium as gym
import minigrid
import numpy as np
from gymnasium.envs.registration import EnvSpec

from rl_practice.envs.color_panel_env import ColorPanelEnv
from rl_practice.envs.four_quadrant_env import FourQuadrantEnv
from rl_practice.envs.letter_tracing_env import LetterTracingEnv
from rl_practice.envs.moving_circle_env import MovingCircleEnv

REPEAT = 4


def _car_racing_action_prompt(horizon: int) -> str:
    base = (
        "You control the red car in CarRacing-v3 (top-down). Stay on the gray road and avoid going onto the green grass; hug the road center when possible. "
        "Action space: steer [-1, +1] where -1 is full left and +1 is full right; accel [-1, +1] where positive is gas and negative is brake. "
        "Typical actions: Turn Left -> steer=-0.20, accel=0.00; Turn Right -> steer=0.20, accel=0.00; Go Straight -> steer=0.00, accel=0.10; Slow Down -> steer=0.00, accel=-0.10. "
    )
    example_actions = "; ".join([f"t{i}: steer=0.00, accel=0.10" for i in range(horizon)])
    return (
        base + f"Respond with {horizon} sequential actions in format: 'Actions: {example_actions}'"
    )


def _car_racing_parse_action(action_text: str, horizon: int) -> tuple[np.ndarray, bool]:
    action_array = np.zeros((horizon, 2), dtype=np.float32)
    pattern = r"(?:t\d+:\s*)?steer=([+-]?\d*\.?\d+),\s*accel=([+-]?\d*\.?\d+)"
    matches = re.findall(pattern, action_text)
    parsed_count = min(len(matches), horizon)
    success = parsed_count == horizon
    for i in range(parsed_count):
        steer = float(matches[i][0])
        accel = float(matches[i][1])
        action_array[i, 0] = np.clip(steer, -1.0, 1.0)
        action_array[i, 1] = np.clip(accel, -1.0, 1.0)
    return action_array, success


def make_env(env_id: str) -> gym.Env:
    if env_id == "MiniGrid-MemoryS9-v0":
        env = gym.make(env_id, agent_view_size=3, render_mode="rgb_array")
        env = minigrid.wrappers.RGBImgPartialObsWrapper(env, tile_size=32)
        env = minigrid.wrappers.ImgObsWrapper(env)
        env = ReduceActionSpaceWrapper(env, n_actions=3)
        env = DiscreteToContinuousWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = TransposeAndNormalizeObs(env)
        env = ZeroObsOnDoneWrapper(env)
        env.unwrapped.spec.reward_threshold = 0.95
        env.unwrapped.eval_range = 100
        return env

    elif env_id == "CarRacing-v3":
        env = gym.make(env_id, render_mode="rgb_array")
        env = env.env  # Unwrap the original TimeLimit wrapper
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000 * REPEAT)
        env = CarRacingRewardFixWrapper(env)
        env = CarRacingActionWrapper(env)
        env = ActionRepeatWrapper(env, repeat=REPEAT)
        env = AverageRewardEarlyStopWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = TransposeAndNormalizeObs(env)
        env = ZeroObsOnDoneWrapper(env)
        env.unwrapped.spec.reward_threshold = 800.0
        env.unwrapped.eval_range = 20
        env.unwrapped.get_action_prompt = _car_racing_action_prompt
        env.unwrapped.parse_action_text = _car_racing_parse_action
        return env

    elif env_id == "CARLA-Leaderboard-v0":
        from rl_practice.envs.carla_leaderboard_env import CARLALeaderboardEnv

        env = CARLALeaderboardEnv()
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = ZeroObsOnDoneWrapper(env)
        env.unwrapped.spec = EnvSpec(id=env_id, reward_threshold=100.0)
        env.unwrapped.eval_range = 100
        return env

    elif env_id == "LetterTracing-v0":
        env = LetterTracingEnv(render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = TransposeAndNormalizeObs(env)
        env = ZeroObsOnDoneWrapper(env)
        env.unwrapped.spec = EnvSpec(id=env_id, reward_threshold=800.0)
        env.unwrapped.eval_range = 20
        return env

    elif env_id == "FourQuadrant-v0":
        env = FourQuadrantEnv(render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = TransposeAndNormalizeObs(env)
        env = ZeroObsOnDoneWrapper(env)
        env.unwrapped.spec = EnvSpec(id=env_id, reward_threshold=800.0)
        env.unwrapped.eval_range = 20
        return env

    elif env_id == "ColorPanel-v0":
        env = ColorPanelEnv(render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = TransposeAndNormalizeObs(env)
        env = ZeroObsOnDoneWrapper(env)
        env.unwrapped.spec = EnvSpec(id=env_id, reward_threshold=800.0)
        env.unwrapped.eval_range = 20
        return env

    elif env_id == "MovingCircle-v0":
        env = MovingCircleEnv(render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = TransposeAndNormalizeObs(env)
        env = ZeroObsOnDoneWrapper(env)
        env.unwrapped.spec = EnvSpec(id=env_id, reward_threshold=800.0)
        env.unwrapped.eval_range = 20
        return env

    elif env_id == "Hopper-v5":
        env = gym.make(env_id, render_mode="rgb_array")
        env = PixelObsWrapper(env, height=96, width=96)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = TransposeAndNormalizeObs(env)
        env = ZeroObsOnDoneWrapper(env)
        env.unwrapped.spec.reward_threshold = 3800.0
        env.unwrapped.eval_range = 20
        return env

    else:
        raise ValueError(f"Unsupported environment: {env_id}")


class DiscreteToContinuousWrapper(gym.Wrapper):
    """
    Convert discrete action space to continuous.
    Example: Discrete(4) -> Box(-1.0, 1.0, (4,))
    The max action is selected.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(env.action_space.n,), dtype=np.float32
        )

    def step(self, action: np.ndarray) -> tuple:
        # Convert continuous action to discrete by selecting the max action
        discrete_action = np.argmax(action)
        return self.env.step(discrete_action)


class ReduceActionSpaceWrapper(gym.Wrapper):
    """
    Reduce discrete action space to only relevant actions.
    For MiniGrid Memory environments, reduce to 3 actions: turn left, turn right, move forward.
    """

    def __init__(self, env: gym.Env, n_actions: int) -> None:
        super().__init__(env)
        self.n_actions = n_actions
        self.action_space = gym.spaces.Discrete(n_actions)

    def step(self, action: np.ndarray | int) -> tuple:
        # Convert action if it's an array
        if isinstance(action, np.ndarray):
            action = action[0] if len(action) > 0 else action
        return self.env.step(action)


class ActionRepeatWrapper(gym.Wrapper):
    """
    Repeat the same action for multiple steps
    """

    def __init__(self, env: gym.Env, repeat: int) -> None:
        super().__init__(env)
        self.repeat = repeat

    def step(self, action: np.ndarray) -> tuple:
        total_reward = 0
        for _ in range(self.repeat):
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

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.window_size = 20
        self.recent_rewards = []

    def reset(self, **kwargs) -> tuple:
        self.recent_rewards = []
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.recent_rewards.append(reward)
        self.recent_rewards = self.recent_rewards[-self.window_size :]

        if len(self.recent_rewards) >= self.window_size:
            count = sum(r < 0.0 for r in self.recent_rewards)
            if count == self.window_size:
                truncated = True

        return obs, reward, terminated, truncated, info


class TransposeAndNormalizeObs(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        h, w = env.observation_space.shape[0:2]
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(3, h, w), dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        o = obs.astype(np.float32) / 255.0
        # Convert from (H, W, C) to (C, H, W)
        o = np.transpose(o, (2, 0, 1))
        return o


class CarRacingRewardFixWrapper(gym.Wrapper):
    """
    Fix CarRacing's -100 penalty for going off-track.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def step(self, action: np.ndarray) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Fix the -100 penalty
        if reward < -30:
            reward += 100

        return obs, reward, terminated, truncated, info


class CarRacingActionWrapper(gym.ActionWrapper):
    """
    Convert 2D action space (steer, gas_or_brake) to 3D action space (steer, gas, brake).
    - steer: [-1, +1] (unchanged)
    - gas_or_brake: [-1, +1]
      - positive: gas=value, brake=0
      - negative: gas=0, brake=abs(value)
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]).astype(np.float32),
            high=np.array([+1.0, +1.0]).astype(np.float32),
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        steer = action[0]
        gas_or_brake = action[1]
        gas_or_brake *= 0.25  # scale down
        gas = np.maximum(gas_or_brake, 0.0)
        brake = np.maximum(-gas_or_brake, 0.0)
        return np.array([steer, gas, brake], dtype=np.float32)


class ResizeObs(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, shape: tuple[int, ...]) -> None:
        super().__init__(env)
        self.shape = shape
        h, w = shape[1:]  # shape is (C, H, W), so extract H, W
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(h, w, 3), dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # obs is (H, W, C), resize and return (H, W, C)
        h, w = self.shape[1:]  # target height and width
        return cv2.resize(obs, (w, h), interpolation=cv2.INTER_AREA)


class PixelObsWrapper(gym.ObservationWrapper):
    """Replace state-vector observation with rendered RGB pixels."""

    def __init__(self, env: gym.Env, height: int, width: int) -> None:
        super().__init__(env)
        self.height = height
        self.width = width
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, 3), dtype=np.uint8
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        img = self.env.render()
        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return img


class ZeroObsOnDoneWrapper(gym.ObservationWrapper):
    """
    Zero out observations when episode is terminated or truncated.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs

    def step(self, action: np.ndarray) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Zero out observation if episode is done
        if terminated or truncated:
            obs = np.zeros_like(obs)

        return obs, reward, terminated, truncated, info
