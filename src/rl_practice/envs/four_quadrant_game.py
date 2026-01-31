# SPDX-License-Identifier: MIT
"""
Four Quadrant Click Game
Divides the screen into 4 quadrants, with 1 quadrant colored red and the rest white
Clicking red -> reward +1, Clicking white -> reward -1
"""

import argparse
import random

import numpy as np
import pygame

# State constants
STATE_PLAYING = "PLAYING"
STATE_SHOW_SCORE = "SHOW_SCORE"
STATE_WAITING = "WAITING"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_limit", type=int)
    return parser.parse_args()


class FourQuadrantGame:
    def __init__(self, time_limit):
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

        # Define 4 quadrant rectangles
        half_w = self.width // 2
        half_h = self.height // 2
        self.quadrants = [
            pygame.Rect(0, 0, half_w, half_h),  # Top-left
            pygame.Rect(half_w, 0, half_w, half_h),  # Top-right
            pygame.Rect(0, half_h, half_w, half_h),  # Bottom-left
            pygame.Rect(half_w, half_h, half_w, half_h),  # Bottom-right
        ]

        # State management
        self.state = STATE_PLAYING
        self.score = 0.0
        self.state_timer = 0
        self.score_duration = 500
        self.waiting_duration = 200

        # Time limit (milliseconds)
        self.time_limit = time_limit if time_limit is not None else 3000
        self.start_time = 0

        # Cooldown period (time to disable judgment after score display, milliseconds)
        self.cooldown_duration = 150
        self.cooldown_end_time = 0

        # Index of the current correct quadrant
        self.correct_quadrant = 0

        # Generate a new question
        self.new_question()

    def new_question(self):
        """Generate a new question"""
        # Randomly select one quadrant
        self.correct_quadrant = random.randint(0, 3)

        # Reset state
        self.state = STATE_PLAYING
        self.score = 0.0
        self.state_timer = 0
        self.start_time = pygame.time.get_ticks()

        # Set cooldown period (wait a bit after score display ends)
        self.cooldown_end_time = pygame.time.get_ticks() + self.cooldown_duration

    def handle_events(self):
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
            self.score = -0.5
            self.state = STATE_SHOW_SCORE
            self.state_timer = pygame.time.get_ticks()
        elif self.state == STATE_PLAYING and mouse_pressed:
            # Clicked in normal state
            pos = pygame.mouse.get_pos()
            self._on_click(pos)

        return True

    def _on_click(self, pos):
        """Process click event"""
        # Determine which quadrant was clicked
        clicked_quadrant = None
        for i, rect in enumerate(self.quadrants):
            if rect.collidepoint(pos):
                clicked_quadrant = i
                break

        # Calculate score
        if clicked_quadrant == self.correct_quadrant:
            self.score = 1.0  # Correct
        else:
            self.score = -0.01  # Incorrect

        # Transition to score display mode
        self.state = STATE_SHOW_SCORE
        self.state_timer = pygame.time.get_ticks()

    def update(self):
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

    def draw(self):
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
            # Draw each quadrant
            for i, rect in enumerate(self.quadrants):
                if i == self.correct_quadrant:
                    # Correct quadrant is red
                    pygame.draw.rect(self.screen, self.RED, rect)
                else:
                    # Others are white (background is already white, so just draw border)
                    pygame.draw.rect(self.screen, self.BLACK, rect, 1)

        pygame.display.flip()

    def _draw_score(self):
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

    def get_screen_array(self):
        """Get screen as numpy array (for Gymnasium environment)"""
        # Convert Pygame screen to numpy array
        array = pygame.surfarray.array3d(self.screen)
        # Transpose (width, height, 3) -> (height, width, 3)
        array = np.transpose(array, (1, 0, 2))
        return array

    def run(self):
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
    game = FourQuadrantGame(args.time_limit)
    game.run()
