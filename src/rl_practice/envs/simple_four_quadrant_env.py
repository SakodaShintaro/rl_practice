"""
Four Quadrant Game - Implemented with numpy without Pygame
Screen divided into 4 quadrants, 1 is red, others are white
Red click -> reward +1, White click -> reward -0.1
"""

import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image, ImageDraw, ImageFont

# State constants
STATE_PLAYING = "PLAYING"
STATE_SHOW_SCORE = "SHOW_SCORE"
STATE_WAITING = "WAITING"


class SimpleFourQuadrantEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode):
        super().__init__()

        self.render_mode = render_mode

        # Fixed values
        self.width = 192
        self.height = 192

        # Color definitions (RGB)
        self.WHITE = np.array([255, 255, 255], dtype=np.uint8)
        self.BLACK = np.array([0, 0, 0], dtype=np.uint8)
        self.RED = np.array([255, 0, 0], dtype=np.uint8)
        self.SCORE_BG = np.array([255, 255, 200], dtype=np.uint8)
        self.SCORE_BORDER = np.array([200, 150, 0], dtype=np.uint8)

        # Define 4 quadrant rectangles (x, y, w, h)
        half_w = self.width // 2
        half_h = self.height // 2
        self.quadrants = [
            (0, 0, half_w, half_h),
            (half_w, 0, half_w, half_h),
            (0, half_h, half_w, half_h),
            (half_w, half_h, half_w, half_h),
        ]

        # Action space: [x, y, button_state]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        # Observation space: RGB image
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 3),
            dtype=np.uint8,
        )

        # Current correct quadrant index
        self.correct_quadrant = 0

        # Step counter
        self.step_count = 0

        # Mouse button state
        self.prev_button_state = False

        # State management
        self.state = STATE_PLAYING
        self.current_score = 0.0
        self.state_timer = 0
        self.score_duration = 3
        self.waiting_duration = 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Generate new problem
        self.correct_quadrant = random.randint(0, 3)
        self.step_count = 0
        self.prev_button_state = False
        self.state = STATE_PLAYING
        self.current_score = 0.0
        self.state_timer = 0

        # Get initial observation
        observation = self._get_observation()

        info = {}
        return observation, info

    def step(self, action):
        # Increment step counter
        self.step_count += 1

        # Interpret action
        x_norm, y_norm, button_state = action

        # Clip action to 0.0-1.0 range
        x_norm = np.clip(x_norm, 0.0, 1.0)
        y_norm = np.clip(y_norm, 0.0, 1.0)

        # Convert to screen coordinates
        x = int(x_norm * (self.width - 1))
        y = int(y_norm * (self.height - 1))

        # Determine button state
        current_button_state = button_state > 0.5

        reward = 0.0

        # Process based on state
        if self.state == STATE_SHOW_SCORE:
            # Processing during score display
            self.state_timer += 1
            if self.state_timer >= self.score_duration:
                self.state = STATE_WAITING
                self.state_timer = 0
        elif self.state == STATE_WAITING:
            # Processing during no-red state
            self.state_timer += 1

            # Click detection (while button is pressed)
            if current_button_state:
                # Penalty for clicking during no-red state
                self.current_score = -0.5
                self.state = STATE_SHOW_SCORE
                self.state_timer = 0
                reward = self.current_score
            elif self.state_timer >= self.waiting_duration:
                # Show red after certain time (new problem)
                self.state = STATE_PLAYING
                self.correct_quadrant = random.randint(0, 3)
        else:
            # Penalty for doing nothing during normal state
            reward = -0.5

            # Click detection (while button is pressed)
            if current_button_state:
                # Determine which quadrant was clicked
                clicked_quadrant = None
                for i, (qx, qy, qw, qh) in enumerate(self.quadrants):
                    if qx <= x < qx + qw and qy <= y < qy + qh:
                        clicked_quadrant = i
                        break

                # Calculate reward
                if clicked_quadrant == self.correct_quadrant:
                    reward = 1.0
                else:
                    reward = -0.01

                # Transition to score display mode
                self.current_score = reward
                self.state = STATE_SHOW_SCORE
                self.state_timer = 0

        # Update button state
        self.prev_button_state = current_button_state

        # Get observation
        observation = self._get_observation()

        # Check termination
        terminated = False
        truncated = self.step_count >= 200

        info = {}

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """Get observation (RGB image of screen)"""
        return self._render_frame()

    def _render_frame(self):
        """Render frame - generate directly with numpy"""
        # Create white background image
        image = np.full((self.height, self.width, 3), self.WHITE, dtype=np.uint8)

        if self.state == STATE_SHOW_SCORE:
            # Show score
            self._draw_score(image)
        elif self.state == STATE_WAITING:
            # No-red state (all white)
            pass
        else:
            # Draw each quadrant
            for i, (x, y, w, h) in enumerate(self.quadrants):
                if i == self.correct_quadrant:
                    # Fill correct quadrant with red
                    image[y : y + h, x : x + w] = self.RED
                else:
                    # Draw border only for others (black lines)
                    image[y, x : x + w] = self.BLACK
                    image[y + h - 1, x : x + w] = self.BLACK
                    image[y : y + h, x] = self.BLACK
                    image[y : y + h, x + w - 1] = self.BLACK

        return image

    def _draw_score(self, image):
        """Display score at center of screen"""
        # Draw with PIL
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        # Rectangle for score display (yellow background)
        score_box_width = 150
        score_box_height = 80
        score_box_x = (self.width - score_box_width) // 2
        score_box_y = (self.height - score_box_height) // 2

        # Background (light yellow)
        draw.rectangle(
            [
                score_box_x,
                score_box_y,
                score_box_x + score_box_width,
                score_box_y + score_box_height,
            ],
            fill=tuple(self.SCORE_BG),
            outline=tuple(self.SCORE_BORDER),
            width=2,
        )

        # Draw text
        score_text = f"{self.current_score:.2f}"

        # Adjust font size and center text
        font_size = 36
        font = ImageFont.load_default()

        # Calculate text position (center aligned)
        text_bbox = draw.textbbox((0, 0), score_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = self.width // 2 - text_width // 2
        text_y = self.height // 2 - text_height // 2

        draw.text((text_x, text_y), score_text, fill=tuple(self.BLACK), font=font)

        # Convert PIL image back to numpy array
        image[:] = np.array(pil_image)

    def render(self):
        """Rendering"""
        return self._render_frame()

    def close(self):
        """Close environment"""
        pass
