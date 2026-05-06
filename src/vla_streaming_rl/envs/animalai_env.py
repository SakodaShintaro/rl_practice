# SPDX-License-Identifier: MIT
"""Animal-AI Gymnasium environment.

Wraps Animal-AI v5 (which exposes a Unity ML-Agents BehaviorSpec interface)
as a single-agent gym.Env. The native AAI action is MultiDiscrete([3, 3]):
    dim 0 (forward/back): 0=noop, 1=forward, 2=back
    dim 1 (rotate):       0=noop, 1=right,   2=left
We expose this as Box(-1, 1, shape=(2,)) for parity with the project's other
environments (CARLA, GUI games), discretising with a +/-1/3 dead-zone.
"""

import random
from pathlib import Path

import gymnasium as gym
import numpy as np
from animalai import AnimalAIEnvironment
from gymnasium import spaces
from mlagents_envs.base_env import ActionTuple


def _to_discrete(value: float) -> int:
    """Map a continuous control in [-1, 1] to AAI's {0=noop, 1, 2}."""
    if value >= 1.0 / 3.0:
        return 1
    elif value <= -1.0 / 3.0:
        return 2
    return 0


class AnimalAIEnv(gym.Env):
    """Animal-AI environment with continuous Box action space."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        binary_path: str,
        arena_yaml: str,
        prompt: str,
        resolution: int,
        max_episode_steps: int,
        seed: int,
        base_port: int,
    ):
        super().__init__()
        self.prompt = prompt

        self.binary_path = binary_path
        self.arena_yaml = arena_yaml
        self.resolution = resolution
        self.max_episode_steps = max_episode_steps
        self.seed_value = seed
        # Add jitter so parallel envs don't fight over the same socket.
        self.base_port = base_port + random.randint(0, 1000)
        self.render_mode = "rgb_array"

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(resolution, resolution, 3), dtype=np.uint8
        )

        self._aai: AnimalAIEnvironment | None = None
        self._behavior_name: str | None = None
        self._latest_image: np.ndarray | None = None
        self.episode_step = 0

    def _ensure_started(self):
        if self._aai is not None:
            return
        self._aai = AnimalAIEnvironment(
            file_name=self.binary_path,
            arenas_configurations=self.arena_yaml,
            seed=self.seed_value,
            play=False,
            useCamera=True,
            resolution=self.resolution,
            useRayCasts=False,
            # `--no-graphics-monitor` enables off-screen rendering on a host
            # without a window manager. `no_graphics=True` (the alternative
            # for headless) disables the renderer entirely and produces a
            # solid-colour image, which is unusable for vision policies.
            no_graphics=False,
            additional_args=["--no-graphics-monitor"],
            base_port=self.base_port,
            inference=False,
            use_YAML=True,
        )
        self._behavior_name = next(iter(self._aai.behavior_specs.keys()))

    def _decode_obs(self, obs_chw_float: np.ndarray) -> np.ndarray:
        # AAI emits float32 in [0, 1] with shape (3, H, W).
        return (obs_chw_float.transpose(1, 2, 0) * 255.0).astype(np.uint8)

    def reset(
        self,
        seed: int | None,
        options: dict | None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._ensure_started()
        self._aai.reset()
        self.episode_step = 0

        decision_steps, _ = self._aai.get_steps(self._behavior_name)
        self._latest_image = self._decode_obs(decision_steps.obs[0][0])
        return self._latest_image, {"task_prompt": self.prompt}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        a = np.asarray(action, dtype=np.float32)
        discrete = np.array(
            [[_to_discrete(float(a[0])), _to_discrete(float(a[1]))]], dtype=np.int32
        )
        action_tuple = ActionTuple(
            continuous=np.zeros((1, 0), dtype=np.float32),
            discrete=discrete,
        )
        self._aai.set_actions(self._behavior_name, action_tuple)
        self._aai.step()

        decision_steps, terminal_steps = self._aai.get_steps(self._behavior_name)
        terminated = len(terminal_steps) > 0
        if terminated:
            reward = float(terminal_steps.reward[0])
            self._latest_image = self._decode_obs(terminal_steps.obs[0][0])
        else:
            reward = float(decision_steps.reward[0])
            self._latest_image = self._decode_obs(decision_steps.obs[0][0])

        self.episode_step += 1
        truncated = self.episode_step >= self.max_episode_steps

        info = {"task_prompt": self.prompt, "episode_step": self.episode_step}
        return self._latest_image, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        return self._latest_image

    def close(self):
        if self._aai is not None:
            self._aai.close()
            self._aai = None


if __name__ == "__main__":
    bin_path = Path.home() / "animalai_env" / "Linux" / "animalAI.x86_64"
    arena_path = Path(__import__("animalai").arenas.__file__).parent / "GoodGoal_Random.yml"
    env = AnimalAIEnv(
        binary_path=str(bin_path),
        arena_yaml=str(arena_path),
        prompt="Find and reach the goal sphere placed at a random location in the arena.",
        resolution=96,
        max_episode_steps=100,
        seed=0,
        base_port=5005,
    )
    obs, info = env.reset(seed=0, options=None)
    print(f"obs shape={obs.shape} dtype={obs.dtype} mean={obs.mean():.1f}")
    print(f"info={info}")
    total_reward = 0.0
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"step={i} terminated={terminated} truncated={truncated} reward={reward:.4f}")
            break
    print(f"total_reward={total_reward:.4f}")
    env.close()
