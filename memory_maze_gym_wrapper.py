from typing import Any, Tuple

import cv2
import gymnasium as gym
import numpy as np
from dm_control import composer
from dm_control.locomotion.arenas import labmaze_textures
from dm_env import specs
from gymnasium import spaces
from memory_maze.maze import (
    FixedFloorTexture,
    FixedWallTexture,
    MazeWithTargetsArena,
    RollingBallWithFriction,
)
from memory_maze.tasks import MemoryMazeTask
from memory_maze.wrappers import (
    AgentPositionWrapper,
    MazeLayoutWrapper,
    RemapObservationWrapper,
    TargetColorAsBorderWrapper,
    TargetsPositionWrapper,
)


class MemoryMazeGymWrapper(gym.Env):
    """
    Gymnasium-compatible wrapper for memory_maze environments.
    This is a modified version of memory_maze's GymWrapper to support gymnasium API.
    """

    def __init__(self, env_id: str):
        # Image sizes
        self.obs_size = 96
        self.render_size = 256

        # Parse the environment ID to get the size
        # Format: memory_maze:MemoryMaze-9x9-v0
        if "9x9" in env_id:
            maze_size, n_targets, time_limit = 9, 3, 250
        elif "11x11" in env_id:
            maze_size, n_targets, time_limit = 11, 4, 500
        elif "13x13" in env_id:
            maze_size, n_targets, time_limit = 13, 5, 750
        elif "15x15" in env_id:
            maze_size, n_targets, time_limit = 15, 6, 1000
        else:
            raise ValueError(f"Unsupported memory_maze environment: {env_id}")

        # Create environment with BOTH cameras enabled
        # This is based on memory_maze/tasks.py but with custom obs mapping
        self.env = self._create_memory_maze_env(maze_size, n_targets, time_limit)

        # Set action space
        self.action_space = _convert_to_space(self.env.action_spec())

        # Agent gets egocentric camera
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8
        )
        self._last_render_image = None

    def _create_memory_maze_env(self, maze_size, n_targets, time_limit):
        """Create memory_maze environment with both egocentric and top cameras."""
        # Based on tasks._memory_maze
        random_state = np.random.RandomState(None)
        walker = RollingBallWithFriction(camera_height=0.3, add_ears=True)
        arena = MazeWithTargetsArena(
            x_cells=maze_size + 2,
            y_cells=maze_size + 2,
            xy_scale=2.0,
            z_height=1.5,
            max_rooms=6,
            room_min_size=3,
            room_max_size=5,
            spawns_per_room=1,
            targets_per_room=1,
            floor_textures=FixedFloorTexture("style_01", ["blue", "blue_bright"]),
            wall_textures=dict(
                {"*": FixedWallTexture("style_01", "yellow")},
                **{str(i): labmaze_textures.WallTextures("style_01") for i in range(10)},
            ),
            skybox_texture=None,
            random_seed=random_state.randint(2147483648),
        )

        task = MemoryMazeTask(
            walker=walker,
            maze_arena=arena,
            n_targets=n_targets,
            target_radius=0.6,
            target_height_above_ground=-0.6,
            enable_global_task_observables=True,
            control_timestep=1.0 / 4.0,
            camera_resolution=self.render_size,
            target_randomize_colors=False,
        )

        # Enable top camera
        task.observables["top_camera"].enabled = True

        env = composer.Environment(
            time_limit=time_limit - 1e-3,
            task=task,
            random_state=random_state,
            strip_singleton_obs_buffer_dim=True,
        )

        # Add position and layout wrappers
        env = TargetsPositionWrapper(env, arena.xy_scale, arena.maze.width, arena.maze.height)
        env = AgentPositionWrapper(env, arena.xy_scale, arena.maze.width, arena.maze.height)
        env = MazeLayoutWrapper(env)

        # Custom observation mapping: keep BOTH cameras
        obs_mapping = {
            "image": "walker/egocentric_camera",  # For TargetColorAsBorderWrapper
            "egocentric_camera": "walker/egocentric_camera",
            "top_camera": "top_camera",
            "target_color": "target_color",
            "agent_pos": "agent_pos",
            "agent_dir": "agent_dir",
            "targets_pos": "targets_pos",
            "target_pos": "target_pos",
            "maze_layout": "maze_layout",
        }
        env = RemapObservationWrapper(env, obs_mapping)
        env = TargetColorAsBorderWrapper(env)  # This modifies 'image' in-place

        return env

    def _extract_observations(self, obs_dict):
        """Extract egocentric camera for agent and top camera for render."""
        # Agent observation: 'image' (egocentric with target color border)
        agent_obs = obs_dict["image"]
        agent_obs = cv2.resize(
            agent_obs, (self.obs_size, self.obs_size), interpolation=cv2.INTER_LINEAR
        )
        return agent_obs, obs_dict["top_camera"]

    def reset(self, seed=None, options=None) -> Tuple[Any, dict]:
        ts = self.env.reset()
        agent_obs, self._last_render_image = self._extract_observations(ts.observation)
        return agent_obs, {}

    def step(self, action) -> Tuple[Any, float, bool, bool, dict]:
        ts = self.env.step(action)
        assert not ts.first(), "dm_env.step() caused reset, reward will be undefined."
        assert ts.reward is not None
        terminal = ts.last() and ts.discount == 0.0
        truncated = ts.last() and ts.discount != 0.0
        agent_obs, self._last_render_image = self._extract_observations(ts.observation)
        return agent_obs, ts.reward, terminal, truncated, {}

    def render(self):
        if self._last_render_image is not None:
            return self._last_render_image
        # Default: black image with render_size
        return np.zeros((self.render_size, self.render_size, 3), dtype=np.uint8)


def _convert_to_space(spec: Any) -> gym.Space:
    # Inverse of acme.gym_wrappers._convert_to_spec

    if isinstance(spec, specs.DiscreteArray):
        return spaces.Discrete(spec.num_values)

    if isinstance(spec, specs.BoundedArray):
        return spaces.Box(
            shape=spec.shape,
            dtype=spec.dtype,
            low=spec.minimum.item() if len(spec.minimum.shape) == 0 else spec.minimum,
            high=spec.maximum.item() if len(spec.maximum.shape) == 0 else spec.maximum,
        )

    if isinstance(spec, specs.Array):
        return spaces.Box(shape=spec.shape, dtype=spec.dtype, low=-np.inf, high=np.inf)

    if isinstance(spec, tuple):
        return spaces.Tuple(_convert_to_space(s) for s in spec)

    if isinstance(spec, dict):
        return spaces.Dict({key: _convert_to_space(value) for key, value in spec.items()})

    raise ValueError(f"Unexpected spec: {spec}")
