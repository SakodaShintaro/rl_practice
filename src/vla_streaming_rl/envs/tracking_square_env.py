# SPDX-License-Identifier: MIT
"""
Tracking Square Game - Gymnasium Environment

Two colored squares move with constant velocity, bouncing off walls.
A language instruction tells the agent which color to track.
Reward is based on cursor distance to the target square's center.
"""

import math
import random

import numpy as np

from vla_streaming_rl.envs.base_gui_env import BaseGUIEnv

COLORS = {
    "RED": (255, 0, 0),
    "GREEN": (0, 200, 0),
    "YELLOW": (255, 255, 0),
    "BLUE": (0, 0, 255),
}
COLOR_NAMES = list(COLORS.keys())


class TrackingSquareEnv(BaseGUIEnv):
    def __init__(self, render_mode):
        super().__init__(render_mode)
        self._window_title = "Tracking Square Game"
        self.task_prompt = ""
        self.square_size = self.width // 3
        self.num_squares = 2

        self.squares = []
        self.target_color_name = ""

    def _create_square(self, color_name):
        s = self.square_size
        angle = random.uniform(0, 2 * math.pi)
        speed = 3.0
        return {
            "x": random.uniform(0, self.width - s),
            "y": random.uniform(0, self.height - s),
            "vx": speed * math.cos(angle),
            "vy": speed * math.sin(angle),
            "color_name": color_name,
            "color": COLORS[color_name],
        }

    def _move_squares(self):
        s = self.square_size
        for sq in self.squares:
            sq["x"] += sq["vx"]
            sq["y"] += sq["vy"]

            if sq["x"] < 0:
                sq["x"] = -sq["x"]
                sq["vx"] = -sq["vx"]
            elif sq["x"] > self.width - s:
                sq["x"] = 2 * (self.width - s) - sq["x"]
                sq["vx"] = -sq["vx"]

            if sq["y"] < 0:
                sq["y"] = -sq["y"]
                sq["vy"] = -sq["vy"]
            elif sq["y"] > self.height - s:
                sq["y"] = 2 * (self.height - s) - sq["y"]
                sq["vy"] = -sq["vy"]

    def _get_target_square(self):
        for sq in self.squares:
            if sq["color_name"] == self.target_color_name:
                return sq
        return self.squares[0]

    def _compute_reward(self, cursor_x, cursor_y):
        target = self._get_target_square()
        s = self.square_size
        center_x = target["x"] + s / 2.0
        center_y = target["y"] + s / 2.0

        dist = math.sqrt((cursor_x - center_x) ** 2 + (cursor_y - center_y) ** 2)

        corners = [(0, 0), (self.width, 0), (0, self.height), (self.width, self.height)]
        max_dist = max(math.sqrt((center_x - cx) ** 2 + (center_y - cy) ** 2) for cx, cy in corners)

        if max_dist == 0:
            return 1.0
        return 1.0 - dist / max_dist

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        chosen = random.sample(COLOR_NAMES, self.num_squares)
        self.squares = [self._create_square(name) for name in chosen]
        self.target_color_name = random.choice(chosen)
        self.task_prompt = f"Track {self.target_color_name}"

        self.step_count = 0
        self.cursor_x = 0.5
        self.cursor_y = 0.5

        if self.render_mode == "human":
            print(f"\n=== {self.task_prompt} ===")

        return self._render_frame(), {"task_prompt": self.task_prompt}

    def step(self, action):
        self.step_count += 1
        dx, dy, _button = action
        self._update_cursor(dx, dy)
        x, y = self._cursor_pixel()

        self._move_squares()

        reward = self._compute_reward(x, y)

        observation = self._render_frame()
        truncated = self.step_count >= 100 if self.render_mode != "human" else False

        if self.render_mode == "human":
            self._render_human(observation)
            print(f"[{self.task_prompt}] reward={reward:.4f}")

        return observation, reward, False, truncated, {"task_prompt": self.task_prompt}

    def _render_frame(self):
        image = np.full((self.height, self.width, 3), 255, dtype=np.uint8)

        s = self.square_size
        for sq in self.squares:
            x0 = int(round(sq["x"]))
            y0 = int(round(sq["y"]))
            x1 = min(x0 + s, self.width)
            y1 = min(y0 + s, self.height)
            x0 = max(x0, 0)
            y0 = max(y0, 0)
            image[y0:y1, x0:x1] = sq["color"]

        self._draw_cursor(image)
        return image


if __name__ == "__main__":
    TrackingSquareEnv(render_mode="human").run()
