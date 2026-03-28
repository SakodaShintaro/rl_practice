# SPDX-License-Identifier: MIT
"""
Tracking Square Game - Gymnasium Environment

Two colored squares move with constant velocity, bouncing off walls.
A language instruction tells the agent which color to track.
Reward is based on cursor distance to the target square's center.

Hard mode uses STL-10 images as square textures instead of solid colors.
"""

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

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
        self.hard_mode = False

        self.squares = []
        self.target_label = ""
        self._stl10_dir = Path.home() / "data/stl-10/train"
        self._stl10_classes = []
        self._stl10_cache = {}

    def _load_stl10_classes(self):
        if not self._stl10_classes:
            self._stl10_classes = sorted([d.name for d in self._stl10_dir.iterdir() if d.is_dir()])
            for cls_name in self._stl10_classes:
                cls_dir = self._stl10_dir / cls_name
                self._stl10_cache[cls_name] = sorted(
                    [p for p in cls_dir.iterdir() if p.suffix == ".png"]
                )

    def _load_stl10_texture(self, class_name):
        img_path = random.choice(self._stl10_cache[class_name])
        img = Image.open(img_path).convert("RGB")
        s = self.square_size
        img = img.resize((s, s), Image.BILINEAR)
        return np.array(img)

    def _create_square(self, label, texture):
        s = self.square_size
        angle = random.uniform(0, 2 * math.pi)
        speed = 3.0
        return {
            "x": random.uniform(0, self.width - s),
            "y": random.uniform(0, self.height - s),
            "vx": speed * math.cos(angle),
            "vy": speed * math.sin(angle),
            "label": label,
            "texture": texture,
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
            if sq["label"] == self.target_label:
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

        if self.hard_mode:
            self._load_stl10_classes()
            chosen = random.sample(self._stl10_classes, self.num_squares)
            self.squares = [
                self._create_square(name, self._load_stl10_texture(name)) for name in chosen
            ]
        else:
            chosen = random.sample(COLOR_NAMES, self.num_squares)
            s = self.square_size
            self.squares = [
                self._create_square(name, np.full((s, s, 3), COLORS[name], dtype=np.uint8))
                for name in chosen
            ]
        self.target_label = random.choice(chosen)
        self.task_prompt = (
            f"You are working on a task to track colored tiles. "
            f"A black cross indicates the cursor position. "
            f"Track the center of the **{self.target_label}** tile."
        )

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
            # clamp to screen bounds
            sx0 = max(x0, 0)
            sy0 = max(y0, 0)
            sx1 = min(x0 + s, self.width)
            sy1 = min(y0 + s, self.height)
            # corresponding region in texture
            tx0 = sx0 - x0
            ty0 = sy0 - y0
            tx1 = tx0 + (sx1 - sx0)
            ty1 = ty0 + (sy1 - sy0)
            image[sy0:sy1, sx0:sx1] = sq["texture"][ty0:ty1, tx0:tx1]

        self._draw_cursor(image)
        return image


if __name__ == "__main__":
    TrackingSquareEnv(render_mode="human").run()
