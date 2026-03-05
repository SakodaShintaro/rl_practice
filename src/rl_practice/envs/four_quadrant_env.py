# SPDX-License-Identifier: MIT
"""
Four Quadrant Game - Gymnasium Environment

Screen divided into 4 quadrants, 1 is red, others are white.
Red click -> reward +1, White click -> reward -0.01
"""

import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rl_practice.envs.base_gui_env import BaseGUIEnv

STATE_PLAYING = "PLAYING"
STATE_SHOW_SCORE = "SHOW_SCORE"


class FourQuadrantEnv(BaseGUIEnv):
    def __init__(self, render_mode="rgb_array"):
        super().__init__(render_mode=render_mode)
        self._window_title = "Four Quadrant Game"

        # Colors (RGB)
        self.WHITE = np.array([255, 255, 255], dtype=np.uint8)
        self.BLACK = np.array([0, 0, 0], dtype=np.uint8)
        self.RED = np.array([255, 0, 0], dtype=np.uint8)
        self.SCORE_BG = np.array([255, 255, 200], dtype=np.uint8)
        self.SCORE_BORDER = np.array([200, 150, 0], dtype=np.uint8)

        # 4 quadrants (x, y, w, h)
        half_w = self.width // 2
        half_h = self.height // 2
        self.quadrants = [
            (0, 0, half_w, half_h),
            (half_w, 0, half_w, half_h),
            (0, half_h, half_w, half_h),
            (half_w, half_h, half_w, half_h),
        ]

        self.correct_quadrant = 0
        self.prev_button_state = False

        self.state = STATE_PLAYING
        self.current_score = 0.0
        self.state_timer = 0
        self.score_duration = 3

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.correct_quadrant = random.randint(0, 3)
        self.step_count = 0
        self.cursor_x = 0.5
        self.cursor_y = 0.5
        self.prev_button_state = False
        self.state = STATE_PLAYING
        self.current_score = 0.0
        self.state_timer = 0
        return self._get_observation(), {}

    def step(self, action):
        self.step_count += 1
        dx, dy, button = action
        self._update_cursor(dx, dy)
        x, y = self._cursor_pixel()
        current_button_state = button > 0.5

        reward = 0.0

        if self.state == STATE_SHOW_SCORE:
            self.state_timer += 1
            if self.state_timer >= self.score_duration:
                self.state = STATE_PLAYING
                self.correct_quadrant = random.randint(0, 3)
                self.state_timer = 0
        else:
            if current_button_state:
                clicked_quadrant = None
                for i, (qx, qy, qw, qh) in enumerate(self.quadrants):
                    if qx <= x < qx + qw and qy <= y < qy + qh:
                        clicked_quadrant = i
                        break
                if clicked_quadrant == self.correct_quadrant:
                    reward = 1.0
                else:
                    reward = -0.01
                self.current_score = reward
                self.state = STATE_SHOW_SCORE
                self.state_timer = 0

        self.prev_button_state = current_button_state
        observation = self._get_observation()
        truncated = self.step_count >= 200

        if self.render_mode == "human":
            self._render_human(observation)

        return observation, reward, False, truncated, {}

    def _get_observation(self):
        return self._render_frame()

    def _render_frame(self):
        image = np.full((self.height, self.width, 3), self.WHITE, dtype=np.uint8)

        if self.state == STATE_SHOW_SCORE:
            self._draw_score(image)
        else:
            for i, (x, y, w, h) in enumerate(self.quadrants):
                if i == self.correct_quadrant:
                    image[y : y + h, x : x + w] = self.RED
                else:
                    image[y, x : x + w] = self.BLACK
                    image[y + h - 1, x : x + w] = self.BLACK
                    image[y : y + h, x] = self.BLACK
                    image[y : y + h, x + w - 1] = self.BLACK

        self._draw_cursor(image)
        return image

    def _draw_score(self, image):
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        box_w, box_h = 150, 80
        box_x = (self.width - box_w) // 2
        box_y = (self.height - box_h) // 2

        draw.rectangle(
            [box_x, box_y, box_x + box_w, box_y + box_h],
            fill=tuple(self.SCORE_BG),
            outline=tuple(self.SCORE_BORDER),
            width=2,
        )

        score_text = f"{self.current_score:.2f}"
        font = ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), score_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_x = self.width // 2 - text_w // 2
        text_y = self.height // 2 - text_h // 2
        draw.text((text_x, text_y), score_text, fill=tuple(self.BLACK), font=font)

        image[:] = np.array(pil_image)


if __name__ == "__main__":
    FourQuadrantEnv().run()
