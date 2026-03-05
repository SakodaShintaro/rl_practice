# SPDX-License-Identifier: MIT
"""Base class for GUI game Gymnasium environments with cursor support."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class BaseGUIEnv(gym.Env):
    """Base environment for GUI games with delta mouse movement.

    Provides:
    - Common action/observation spaces (192x192 RGB, delta mouse)
    - Cursor position tracking
    - Cursor rendering (crosshair)
    - Pygame human-mode display
    - Human play mode via run()
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode):
        super().__init__()
        self.render_mode = render_mode
        self.width = 192
        self.height = 192

        # Action: (dx, dy, button) ∈ [-1, 1]³
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        # Observation: RGB image
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )

        self.cursor_x = 0.5
        self.cursor_y = 0.5
        self.step_count = 0

        self._pygame_screen = None
        self._window_title = "GUI Game"

    def _update_cursor(self, dx, dy):
        """Update cursor position from delta action."""
        self.cursor_x = np.clip(self.cursor_x + dx * 0.5, 0.0, 1.0)
        self.cursor_y = np.clip(self.cursor_y + dy * 0.5, 0.0, 1.0)

    def _cursor_pixel(self):
        """Get cursor position in pixel coordinates."""
        return (
            int(self.cursor_x * (self.width - 1)),
            int(self.cursor_y * (self.height - 1)),
        )

    def _draw_cursor(self, image):
        """Draw crosshair cursor on image."""
        cx, cy = self._cursor_pixel()
        size = 5
        color = np.array([0, 0, 0], dtype=np.uint8)
        # Horizontal line
        x_start = max(0, cx - size)
        x_end = min(self.width, cx + size + 1)
        image[max(0, cy), x_start:x_end] = color
        # Vertical line
        y_start = max(0, cy - size)
        y_end = min(self.height, cy + size + 1)
        image[y_start:y_end, max(0, cx)] = color

    def _render_human(self, frame):
        """Display frame in Pygame window."""
        try:
            import pygame
        except ImportError:
            return

        if self._pygame_screen is None:
            pygame.init()
            self._pygame_screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption(self._window_title)
            self._pygame_clock = pygame.time.Clock()

        surface = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        self._pygame_screen.blit(surface, (0, 0))
        pygame.display.flip()
        self._pygame_clock.tick(self.metadata["render_fps"])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self._pygame_screen = None

    def render(self):
        frame = self._render_frame()
        if self.render_mode == "human":
            self._render_human(frame)
        return frame

    def close(self):
        if self._pygame_screen is not None:
            import pygame

            pygame.quit()
            self._pygame_screen = None

    def _render_frame(self):
        raise NotImplementedError

    def run(self):
        """Run as standalone GUI game with Pygame mouse input.

        Translates real mouse position to cursor delta actions each frame.
        Left mouse button maps to button action.
        """
        import pygame

        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self._window_title)
        clock = pygame.time.Clock()

        self.reset()
        prev_mouse_x, prev_mouse_y = self.width // 2, self.height // 2
        pygame.mouse.set_pos(prev_mouse_x, prev_mouse_y)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Get mouse state
            mouse_x, mouse_y = pygame.mouse.get_pos()
            mouse_pressed = pygame.mouse.get_pressed()[0]

            # Convert mouse position delta to action delta (normalized)
            # ±1.0 in action space = ±0.5 in cursor space, so scale accordingly
            dx_pixels = mouse_x - prev_mouse_x
            dy_pixels = mouse_y - prev_mouse_y
            dx = dx_pixels / (self.width * 0.5)  # pixels -> action scale
            dy = dy_pixels / (self.height * 0.5)
            dx = np.clip(dx, -1.0, 1.0)
            dy = np.clip(dy, -1.0, 1.0)
            button = 1.0 if mouse_pressed else 0.0

            prev_mouse_x, prev_mouse_y = mouse_x, mouse_y

            action = np.array([dx, dy, button], dtype=np.float32)
            obs, reward, terminated, truncated, info = self.step(action)

            if terminated or truncated:
                self.reset()
                prev_mouse_x, prev_mouse_y = self.width // 2, self.height // 2
                pygame.mouse.set_pos(prev_mouse_x, prev_mouse_y)

            # Display observation
            surface = pygame.surfarray.make_surface(obs.transpose(1, 0, 2))
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            clock.tick(self.metadata["render_fps"])

        pygame.quit()
