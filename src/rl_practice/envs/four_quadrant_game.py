# SPDX-License-Identifier: MIT
"""
Red Rectangle Click Game
Places a red rectangle (half screen size) at a random position.
Clicking red -> reward +1, Clicking white -> reward -0.01
"""

import argparse
import random

import numpy as np
import pygame

# State constants
STATE_PLAYING = "PLAYING"
STATE_SHOW_SCORE = "SHOW_SCORE"
STATE_WAITING = "WAITING"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_limit_sec", type=int, default=10)
    return parser.parse_args()


class FourQuadrantGame:
    def __init__(self, time_limit_sec: int) -> None:
        pygame.init()

        # Fixed values
        self.width = 192
        self.height = 192

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Four Quadrant Game")

        self.clock = pygame.time.Clock()

        # Color definitions
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.SCORE_BG = (255, 255, 200)
        self.SCORE_BORDER = (200, 150, 0)

        # Red rectangle size (half of screen)
        self.rect_w = self.width // 2
        self.rect_h = self.height // 2

        # Red rectangle position (set in new_question)
        self.red_rect = pygame.Rect(0, 0, self.rect_w, self.rect_h)

        # State management
        self.state = STATE_PLAYING
        self.score = 0.0
        self.state_timer = 0
        self.score_duration = 250
        self.waiting_duration = 200

        # Time limit (milliseconds)
        self.time_limit_msec = time_limit_sec * 1000
        self.start_time = 0

        # Cooldown period (time to disable judgment after score display, milliseconds)
        self.cooldown_duration = 150
        self.cooldown_end_time = 0

        # Generate a new question
        self.new_question()

    def new_question(self) -> None:
        """Generate a new question"""
        # Random top-left position: x in [0, W/2], y in [0, H/2]
        rx = random.randint(0, self.width // 2)
        ry = random.randint(0, self.height // 2)
        self.red_rect = pygame.Rect(rx, ry, self.rect_w, self.rect_h)

        # Reset state
        self.state = STATE_PLAYING
        self.score = 0.0
        self.state_timer = 0
        self.start_time = pygame.time.get_ticks()

        # Set cooldown period (wait a bit after score display ends)
        self.cooldown_end_time = pygame.time.get_ticks() + self.cooldown_duration

    def handle_events(self) -> bool:
        """Event processing"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        # Do not judge during cooldown period
        current_time = pygame.time.get_ticks()
        if current_time < self.cooldown_end_time:
            return True

        # Get mouse button state
        mouse_pressed = pygame.mouse.get_pressed()[0]

        # Process based on state
        if self.state == STATE_WAITING and mouse_pressed:
            # Clicked in no-red state
            self.score = -0.1
            self.state = STATE_SHOW_SCORE
            self.state_timer = pygame.time.get_ticks()
        elif self.state == STATE_PLAYING and mouse_pressed:
            # Clicked in normal state
            pos = pygame.mouse.get_pos()
            self._on_click(pos)

        return True

    def _on_click(self, pos: tuple[int, int]) -> None:
        """Process click event"""
        # Distance from click to nearest point on red rectangle
        nearest_x = max(self.red_rect.left, min(pos[0], self.red_rect.right))
        nearest_y = max(self.red_rect.top, min(pos[1], self.red_rect.bottom))
        dist = ((pos[0] - nearest_x) ** 2 + (pos[1] - nearest_y) ** 2) ** 0.5
        max_dist = ((self.width / 2) ** 2 + (self.height / 2) ** 2) ** 0.5
        normalized_dist = dist / max_dist
        self.score = 1.0 if self.red_rect.collidepoint(pos) else 0.5 / (1 + 10 * normalized_dist)

        # Transition to score display mode
        self.state = STATE_SHOW_SCORE
        self.state_timer = pygame.time.get_ticks()

    def _on_timeout(self) -> None:
        """Process when time limit is reached"""
        self.score = -1.0
        self.state = STATE_SHOW_SCORE
        self.state_timer = pygame.time.get_ticks()

    def update(self) -> None:
        """Update game state"""
        current_time = pygame.time.get_ticks()

        if self.state == STATE_SHOW_SCORE:
            # While showing score
            if current_time - self.state_timer > self.score_duration:
                self.state = STATE_WAITING
                self.state_timer = current_time
        elif self.state == STATE_WAITING:
            # In no-red state
            if current_time - self.state_timer > self.waiting_duration:
                self.new_question()
        elif self.state == STATE_PLAYING:
            # Time limit check
            if current_time - self.start_time > self.time_limit_msec:
                self._on_timeout()

    def draw(self) -> None:
        """Draw screen"""
        # Fill background with white
        self.screen.fill(self.WHITE)

        if self.state == STATE_SHOW_SCORE:
            # Display score
            self._draw_score()
        elif self.state == STATE_WAITING:
            # No-red state (all white)
            pass
        else:
            # Draw red rectangle at random position
            pygame.draw.rect(self.screen, self.RED, self.red_rect)

        pygame.display.flip()

    def _draw_score(self) -> None:
        """Display score at screen center"""
        # Rectangle for score display (yellow background)
        score_box_width = 150
        score_box_height = 80
        score_box_x = (self.width - score_box_width) // 2
        score_box_y = (self.height - score_box_height) // 2

        score_box_rect = pygame.Rect(score_box_x, score_box_y, score_box_width, score_box_height)

        # Background (light yellow)
        pygame.draw.rect(self.screen, self.SCORE_BG, score_box_rect)

        # Border (orange)
        pygame.draw.rect(self.screen, self.SCORE_BORDER, score_box_rect, 2)

        # "Score:" label
        label_font = pygame.font.Font(None, 24)
        label_text = label_font.render("Score:", True, self.BLACK)
        label_rect = label_text.get_rect(center=(self.width // 2, self.height // 2 - 15))
        self.screen.blit(label_text, label_rect)

        # Score value
        score_font = pygame.font.Font(None, 36)
        score_text = f"{self.score:.2f}"
        score_text_surface = score_font.render(score_text, True, self.BLACK)
        score_text_rect = score_text_surface.get_rect(
            center=(self.width // 2, self.height // 2 + 15)
        )
        self.screen.blit(score_text_surface, score_text_rect)

    def get_screen_array(self) -> np.ndarray:
        """Get screen as numpy array (for Gymnasium environment)"""
        # Convert Pygame screen to numpy array
        array = pygame.surfarray.array3d(self.screen)
        # Transpose (width, height, 3) -> (height, width, 3)
        array = np.transpose(array, (1, 0, 2))
        return array

    def run(self) -> None:
        """Game loop"""
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    args = parse_args()
    game = FourQuadrantGame(args.time_limit_sec)
    game.run()
