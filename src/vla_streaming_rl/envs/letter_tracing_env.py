# SPDX-License-Identifier: MIT
"""
Letter Tracing Game - Gymnasium Environment

Displays a-z letters in light gray, agent traces the shape by dragging.
Score is calculated using IoU (Intersection over Union).
button > 0.0 = pen down (drawing)
"""

import random
import string

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vla_streaming_rl.envs.base_gui_env import BaseGUIEnv

STATE_PLAYING = "PLAYING"
STATE_SHOW_SCORE = "SHOW_SCORE"


class LetterTracingEnv(BaseGUIEnv):
    def __init__(self, render_mode):
        super().__init__(render_mode)
        self._window_title = "Letter Tracing Game"
        self.prompt = "A letter is displayed on screen. Trace the letter shape by moving the cursor along its outline."

        self.font_size = 150
        self.sequential = False
        self.current_index = 0

        # Load font once
        try:
            self._font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", self.font_size
            )
        except Exception:
            self._font = ImageFont.load_default()
        self._small_font = ImageFont.load_default()

        self.current_letter = "a"
        self.letter_offset_x = 0
        self.letter_offset_y = 0

        self._letter_mask = None
        self._user_mask = None

        self.prev_button_state = False

        self.state = STATE_PLAYING
        self.current_score = 0.0
        self.state_timer = 0
        self.score_duration = 3
        self.time_limit_steps = 150
        self.brush_radius = 12

    def _new_letter(self):
        if self.sequential:
            self.current_letter = string.ascii_lowercase[self.current_index]
            self.current_index = (self.current_index + 1) % 26
        else:
            self.current_letter = random.choice(string.ascii_lowercase)

        self.letter_offset_x = random.randint(-5, 5)
        self.letter_offset_y = random.randint(-5, 5)

        letter_image = Image.new("RGB", (self.width, self.height), (255, 255, 255))
        draw = ImageDraw.Draw(letter_image)

        bbox = draw.textbbox((0, 0), self.current_letter, font=self._font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (self.width - text_w) // 2 + self.letter_offset_x
        y = (self.height - text_h) // 2 + self.letter_offset_y

        draw.text((x, y), self.current_letter, fill=(200, 200, 200), font=self._font)

        letter_array = np.array(letter_image)
        self._letter_mask = np.any(letter_array != [255, 255, 255], axis=2)
        self._letter_image = letter_array

        self._user_mask = np.zeros((self.height, self.width), dtype=bool)

    def _draw_brush(self, px, py):
        r = self.brush_radius
        y_min = max(0, py - r)
        y_max = min(self.height, py + r + 1)
        x_min = max(0, px - r)
        x_max = min(self.width, px + r + 1)

        for yy in range(y_min, y_max):
            for xx in range(x_min, x_max):
                if (xx - px) ** 2 + (yy - py) ** 2 <= r * r:
                    self._user_mask[yy, xx] = True

    def _calculate_iou(self):
        intersection = np.sum(self._letter_mask & self._user_mask)
        union = np.sum(self._letter_mask | self._user_mask)
        if union == 0:
            return 0.0
        return float(intersection / union)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._new_letter()
        self.step_count = 0
        self.cursor_x = 0.5
        self.cursor_y = 0.5
        self.prev_button_state = False
        self.state = STATE_PLAYING
        self.current_score = 0.0
        self.state_timer = 0
        return self._get_observation(), {"task_prompt": self.prompt}

    def step(self, action):
        self.step_count += 1
        dx, dy, button = action
        self._update_cursor(dx, dy)
        px, py = self._cursor_pixel()
        current_button_state = button > 0.0

        reward = 0.0

        if self.state == STATE_SHOW_SCORE:
            self.state_timer += 1
            if self.state_timer >= self.score_duration:
                self._new_letter()
                self.state = STATE_PLAYING
                self.state_timer = 0
        elif self.state == STATE_PLAYING:
            if current_button_state:
                self._draw_brush(px, py)

            if self.state_timer >= self.time_limit_steps:
                self.current_score = self._calculate_iou()
                reward = self.current_score
                self.state = STATE_SHOW_SCORE
                self.state_timer = 0
            else:
                self.state_timer += 1

        self.prev_button_state = current_button_state
        observation = self._get_observation()
        truncated = self.step_count >= 200

        if self.render_mode == "human":
            self._render_human(observation)

        return observation, reward, False, truncated, {"task_prompt": self.prompt}

    def _get_observation(self):
        return self._render_frame()

    def _render_frame(self):
        if self.state == STATE_SHOW_SCORE:
            image = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
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
            text_bbox = draw.textbbox((0, 0), score_text, font=self._small_font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            draw.text(
                (self.width // 2 - text_w // 2, self.height // 2 - text_h // 2),
                score_text,
                fill=(0, 0, 0),
                font=self._small_font,
            )
            image = np.array(pil_image)
        else:
            image = self._letter_image.copy()
            image[self._user_mask] = [0, 0, 0]

        self._draw_cursor(image)
        return image


if __name__ == "__main__":
    LetterTracingEnv(render_mode="human").run()
