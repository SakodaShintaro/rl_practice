# SPDX-License-Identifier: MIT
"""
Color Panel Click Game
Divides the screen into 4 quadrants, each filled with a different color (red, green, yellow, blue).
A text instruction tells the player which color to click.
Clicking correct color -> reward +1, Clicking wrong color -> reward -0.01
Color positions are randomized each round.
"""

import argparse
import random

import numpy as np
import pygame

# State constants
STATE_PLAYING = "PLAYING"
STATE_SHOW_SCORE = "SHOW_SCORE"
STATE_WAITING = "WAITING"

# Color definitions
COLORS = {
    "RED": (255, 0, 0),
    "GREEN": (0, 200, 0),
    "YELLOW": (255, 255, 0),
    "BLUE": (0, 0, 255),
}
COLOR_NAMES = list(COLORS.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_limit_sec", type=int, default=5)
    return parser.parse_args()


class ColorPanelGame:
    def __init__(self, time_limit_sec: int) -> None:
        pygame.init()

        # Fixed values
        self.width = 192
        self.height = 192

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Color Panel Game")

        self.clock = pygame.time.Clock()

        # Color definitions
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.SCORE_BG = (255, 255, 200)
        self.SCORE_BORDER = (200, 150, 0)
        self.INSTRUCTION_BG = (0, 0, 0)
        self.INSTRUCTION_FG = (255, 255, 255)

        # Define 4 quadrant rectangles
        half_w = self.width // 2
        half_h = self.height // 2
        self.quadrants = [
            pygame.Rect(0, 0, half_w, half_h),  # Top-left
            pygame.Rect(half_w, 0, half_w, half_h),  # Top-right
            pygame.Rect(0, half_h, half_w, half_h),  # Bottom-left
            pygame.Rect(half_w, half_h, half_w, half_h),  # Bottom-right
        ]

        # Color assignment to quadrants (shuffled each round)
        self.color_assignment = list(range(4))  # indices into COLOR_NAMES

        # State management
        self.state = STATE_PLAYING
        self.score = 0.0
        self.state_timer = 0
        self.score_duration = 500
        self.waiting_duration = 200

        # Time limit (milliseconds)
        self.time_limit_msec = time_limit_sec * 1000
        self.start_time = 0

        # Cooldown period (time to disable judgment after score display, milliseconds)
        self.cooldown_duration = 150
        self.cooldown_end_time = 0

        # Index of the correct color (into COLOR_NAMES)
        self.correct_color_idx = 0

        # Font for instruction text
        self.instruction_font = pygame.font.Font(None, 28)

        # Generate a new question
        self.new_question()

    def new_question(self) -> None:
        """Generate a new question"""
        # Shuffle color positions
        random.shuffle(self.color_assignment)

        # Randomly select a target color
        self.correct_color_idx = random.randint(0, 3)

        # Reset state
        self.state = STATE_PLAYING
        self.score = 0.0
        self.state_timer = 0
        self.start_time = pygame.time.get_ticks()

        # Set cooldown period
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
            # Clicked in waiting state
            self.score = -0.5
            self.state = STATE_SHOW_SCORE
            self.state_timer = pygame.time.get_ticks()
        elif self.state == STATE_PLAYING and mouse_pressed:
            # Clicked in normal state
            pos = pygame.mouse.get_pos()
            self._on_click(pos)

        return True

    def _on_click(self, pos: tuple[int, int]) -> None:
        """Process click event"""
        # Determine which quadrant was clicked
        clicked_quadrant = None
        for i, rect in enumerate(self.quadrants):
            if rect.collidepoint(pos):
                clicked_quadrant = i
                break

        if clicked_quadrant is None:
            return

        # Check if the clicked color matches the target
        clicked_color_idx = self.color_assignment[clicked_quadrant]
        if clicked_color_idx == self.correct_color_idx:
            self.score = 1.0  # Correct
        else:
            self.score = -0.01  # Incorrect

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
            # In waiting state
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
            # Waiting state (all white)
            pass
        else:
            # Draw each quadrant with assigned color
            for i, rect in enumerate(self.quadrants):
                color_idx = self.color_assignment[i]
                color_name = COLOR_NAMES[color_idx]
                color = COLORS[color_name]
                pygame.draw.rect(self.screen, color, rect)
                # Draw border
                pygame.draw.rect(self.screen, self.BLACK, rect, 1)

            # Draw instruction text at the top
            self._draw_instruction()

        pygame.display.flip()

    def _draw_instruction(self) -> None:
        """Draw instruction text at the top of the screen"""
        target_name = COLOR_NAMES[self.correct_color_idx]
        text = f"Click {target_name}"

        text_surface = self.instruction_font.render(text, True, self.INSTRUCTION_FG)
        text_rect = text_surface.get_rect(center=(self.width // 2, 16))

        # Draw background rectangle
        bg_rect = text_rect.inflate(12, 6)
        pygame.draw.rect(self.screen, self.INSTRUCTION_BG, bg_rect)
        self.screen.blit(text_surface, text_rect)

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
    game = ColorPanelGame(args.time_limit_sec)
    game.run()
