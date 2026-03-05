# SPDX-License-Identifier: MIT
"""
Color Panel Game - Gymnasium Environment

4 colored quadrants (red, green, yellow, blue) with text instruction.
Click the correct color -> reward +1, wrong -> reward -0.01
"""

import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rl_practice.envs.base_gui_env import BaseGUIEnv

STATE_PLAYING = "PLAYING"
STATE_SHOW_SCORE = "SHOW_SCORE"

COLORS = {
    "RED": (255, 0, 0),
    "GREEN": (0, 200, 0),
    "YELLOW": (255, 255, 0),
    "BLUE": (0, 0, 255),
}
COLOR_NAMES = list(COLORS.keys())


class ColorPanelEnv(BaseGUIEnv):
    def __init__(self, render_mode="rgb_array"):
        super().__init__(render_mode=render_mode)
        self._window_title = "Color Panel Game"

        half_w = self.width // 2
        half_h = self.height // 2
        self.quadrants = [
            (0, 0, half_w, half_h),
            (half_w, 0, half_w, half_h),
            (0, half_h, half_w, half_h),
            (half_w, half_h, half_w, half_h),
        ]

        self.color_assignment = list(range(4))
        self.correct_color_idx = 0

        self.state = STATE_PLAYING
        self.current_score = 0.0
        self.state_timer = 0
        self.score_duration = 3

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        random.shuffle(self.color_assignment)
        self.correct_color_idx = random.randint(0, 3)
        self.step_count = 0
        self.cursor_x = 0.5
        self.cursor_y = 0.5
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
                random.shuffle(self.color_assignment)
                self.correct_color_idx = random.randint(0, 3)
                self.state_timer = 0
        else:
            if current_button_state:
                clicked_quadrant = None
                for i, (qx, qy, qw, qh) in enumerate(self.quadrants):
                    if qx <= x < qx + qw and qy <= y < qy + qh:
                        clicked_quadrant = i
                        break

                if clicked_quadrant is not None:
                    clicked_color_idx = self.color_assignment[clicked_quadrant]
                    if clicked_color_idx == self.correct_color_idx:
                        reward = 1.0
                    else:
                        reward = -0.01

                self.current_score = reward
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

        if self.state == STATE_SHOW_SCORE:
            self._draw_score(image)
        else:
            for i, (qx, qy, qw, qh) in enumerate(self.quadrants):
                color_idx = self.color_assignment[i]
                color_name = COLOR_NAMES[color_idx]
                color = COLORS[color_name]
                image[qy : qy + qh, qx : qx + qw] = color
                # Draw border (black)
                image[qy, qx : qx + qw] = (0, 0, 0)
                image[qy + qh - 1, qx : qx + qw] = (0, 0, 0)
                image[qy : qy + qh, qx] = (0, 0, 0)
                image[qy : qy + qh, qx + qw - 1] = (0, 0, 0)

            self._draw_instruction(image)

        self._draw_cursor(image)
        return image

    def _draw_instruction(self, image):
        target_name = COLOR_NAMES[self.correct_color_idx]
        text = f"Click {target_name}"

        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        font = ImageFont.load_default()

        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        text_x = self.width // 2 - text_w // 2
        text_y = 16 - text_h // 2

        pad_x, pad_y = 6, 3
        draw.rectangle(
            [text_x - pad_x, text_y - pad_y, text_x + text_w + pad_x, text_y + text_h + pad_y],
            fill=(0, 0, 0),
        )
        draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)

        image[:] = np.array(pil_image)

    def _draw_score(self, image):
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        box_w, box_h = 150, 80
        box_x = (self.width - box_w) // 2
        box_y = (self.height - box_h) // 2

        draw.rectangle(
            [box_x, box_y, box_x + box_w, box_y + box_h],
            fill=(255, 255, 200),
            outline=(200, 150, 0),
            width=2,
        )

        score_text = f"{self.current_score:.2f}"
        font = ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), score_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_x = self.width // 2 - text_w // 2
        text_y = self.height // 2 - text_h // 2
        draw.text((text_x, text_y), score_text, fill=(0, 0, 0), font=font)

        image[:] = np.array(pil_image)


if __name__ == "__main__":
    ColorPanelEnv().run()
