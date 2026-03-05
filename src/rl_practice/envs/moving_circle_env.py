# SPDX-License-Identifier: MIT
"""
Moving Circle Game - Gymnasium Environment

Three types of circles move around the screen:
- Green: large, slow, reward +0.1
- Yellow: medium, medium speed, reward +0.5
- Red: small, fast, reward +1.0
"""

import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rl_practice.envs.base_gui_env import BaseGUIEnv

STATE_PLAYING = "PLAYING"
STATE_SHOW_SCORE = "SHOW_SCORE"

CIRCLE_TYPES = {
    "green": {
        "color": (0, 200, 0),
        "radius": 25,
        "speed": 1.0,
        "reward": 0.1,
        "min_steps": 60,
        "max_steps": 120,
    },
    "yellow": {
        "color": (255, 255, 0),
        "radius": 15,
        "speed": 2.0,
        "reward": 0.5,
        "min_steps": 40,
        "max_steps": 80,
    },
    "red": {
        "color": (255, 0, 0),
        "radius": 8,
        "speed": 3.5,
        "reward": 1.0,
        "min_steps": 20,
        "max_steps": 50,
    },
}


class MovingCircleEnv(BaseGUIEnv):
    def __init__(self, render_mode):
        super().__init__(render_mode)
        self._window_title = "Moving Circle Game"

        self.num_circles = {"green": 1, "yellow": 1, "red": 1}

        self.circles = []
        self.state = STATE_PLAYING
        self.current_score = 0.0
        self.total_score = 0.0
        self.state_timer = 0
        self.score_duration = 3

    def _create_circle(self, config):
        radius = config["radius"]
        return {
            "x": random.uniform(radius, self.width - radius),
            "y": random.uniform(radius, self.height - radius),
            "radius": radius,
            "color": config["color"],
            "speed": config["speed"],
            "reward": config["reward"],
            "direction": random.uniform(0, 2 * math.pi),
            "steps_remaining": random.randint(config["min_steps"], config["max_steps"]),
            "min_steps": config["min_steps"],
            "max_steps": config["max_steps"],
        }

    def _respawn_circle(self, circle):
        circle["x"] = random.uniform(circle["radius"], self.width - circle["radius"])
        circle["y"] = random.uniform(circle["radius"], self.height - circle["radius"])
        circle["direction"] = random.uniform(0, 2 * math.pi)
        circle["steps_remaining"] = random.randint(circle["min_steps"], circle["max_steps"])

    def _move_circles(self):
        for circle in self.circles:
            circle["x"] += circle["speed"] * math.cos(circle["direction"])
            circle["y"] += circle["speed"] * math.sin(circle["direction"])

            r = circle["radius"]
            if circle["x"] < r or circle["x"] > self.width - r:
                circle["direction"] = math.pi - circle["direction"]
                circle["x"] = max(r, min(self.width - r, circle["x"]))
            if circle["y"] < r or circle["y"] > self.height - r:
                circle["direction"] = -circle["direction"]
                circle["y"] = max(r, min(self.height - r, circle["y"]))

            circle["steps_remaining"] -= 1
            if circle["steps_remaining"] <= 0:
                circle["direction"] = random.uniform(0, 2 * math.pi)
                circle["steps_remaining"] = random.randint(
                    circle["min_steps"], circle["max_steps"]
                )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.circles = []
        for name, config in CIRCLE_TYPES.items():
            for _ in range(self.num_circles[name]):
                self.circles.append(self._create_circle(config))

        self.step_count = 0
        self.cursor_x = 0.5
        self.cursor_y = 0.5
        self.state = STATE_PLAYING
        self.current_score = 0.0
        self.total_score = 0.0
        self.state_timer = 0
        return self._get_observation(), {}

    def step(self, action):
        self.step_count += 1
        dx, dy, button = action
        self._update_cursor(dx, dy)
        x, y = self._cursor_pixel()
        current_button_state = button > 0.5

        self._move_circles()

        reward = 0.0

        if self.state == STATE_SHOW_SCORE:
            self.state_timer += 1
            if self.state_timer >= self.score_duration:
                self.state = STATE_PLAYING
        elif self.state == STATE_PLAYING and current_button_state:
            clicked_circle = None
            best_reward = -1.0
            for circle in self.circles:
                cdx = x - circle["x"]
                cdy = y - circle["y"]
                if cdx * cdx + cdy * cdy <= circle["radius"] ** 2:
                    if circle["reward"] > best_reward:
                        clicked_circle = circle
                        best_reward = circle["reward"]

            reward = max(0.0, best_reward)
            self.current_score = reward
            self.total_score += reward
            if clicked_circle is not None:
                self._respawn_circle(clicked_circle)
            self.state = STATE_SHOW_SCORE
            self.state_timer = 0

        observation = self._get_observation()
        truncated = self.step_count >= 200

        if self.render_mode == "human":
            self._render_human(observation)

        return observation, reward, False, truncated, {}

    def _get_observation(self):
        return self._render_frame()

    def _render_frame(self):
        image = np.full((self.height, self.width, 3), 255, dtype=np.uint8)

        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        for circle in self.circles:
            cx, cy, r = int(circle["x"]), int(circle["y"]), circle["radius"]
            draw.ellipse(
                [cx - r, cy - r, cx + r, cy + r],
                fill=circle["color"],
            )

        font = ImageFont.load_default()
        draw.text((5, 5), f"Total: {self.total_score:.1f}", fill=(0, 0, 0), font=font)

        if self.state == STATE_SHOW_SCORE:
            box_w, box_h = 120, 60
            box_x = (self.width - box_w) // 2
            box_y = (self.height - box_h) // 2
            draw.rectangle(
                [box_x, box_y, box_x + box_w, box_y + box_h],
                fill=(255, 255, 200),
                outline=(200, 150, 0),
                width=2,
            )
            score_text = f"{self.current_score:+.1f}"
            text_bbox = draw.textbbox((0, 0), score_text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            text_x = self.width // 2 - text_w // 2
            text_y = self.height // 2 - text_h // 2
            draw.text((text_x, text_y), score_text, fill=(0, 0, 0), font=font)

        image = np.array(pil_image)
        self._draw_cursor(image)
        return image


if __name__ == "__main__":
    MovingCircleEnv(render_mode="human").run()
