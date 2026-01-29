"""
Letter Tracing Game
Displays a-z letters in light gray, and the player traces the same shape by mouse dragging
"""

import argparse
import random
import string

import numpy as np
import pygame
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["sequential", "random"], default="random")
    parser.add_argument("--show_done_button", action="store_true")
    return parser.parse_args()


class LetterTracingGame:
    def __init__(self, sequential, show_done_button):
        pygame.init()

        # Fixed values
        self.width = 192
        self.height = 192
        self.font_size = 150

        # Done button display flag
        self.show_done_button = show_done_button

        # Letter display mode
        self.sequential = sequential
        self.current_index = 0  # For sequential mode

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Letter Tracing Game")

        self.clock = pygame.time.Clock()

        # Color definitions
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (200, 200, 200)
        self.BUTTON_COLOR = (100, 100, 200)
        self.BUTTON_HOVER = (150, 150, 255)
        self.SCORE_BG = (255, 255, 200)  # Light yellow (score background)
        self.SCORE_BORDER = (200, 150, 0)  # Orange (score border)

        # Game state
        self.current_letter = None
        self.letter_surface = None
        self.user_surface = None
        self.drawing = False
        self.last_pos = None

        # Score display
        self.show_score = False
        self.score = 0.0
        self.score_timer = 0

        # Time limit (milliseconds)
        self.time_limit = 5000
        self.start_time = 0

        # Done button
        button_width = 50
        button_height = 20
        self.button_rect = pygame.Rect(
            self.width - button_width - 5,
            self.height - button_height - 5,
            button_width,
            button_height,
        )

        # Letter position offset
        self.letter_offset_x = 0
        self.letter_offset_y = 0

        # Generate a new letter
        self.new_letter()

    def new_letter(self):
        """Generate a new letter (sequential or random)"""
        if self.sequential:
            # Sequential mode: a -> b -> c -> ... -> z -> a -> ...
            self.current_letter = string.ascii_lowercase[self.current_index]
            self.current_index = (self.current_index + 1) % 26
        else:
            # Random mode
            self.current_letter = random.choice(string.ascii_lowercase)

        # Random offset (-5 to +5 pixels)
        self.letter_offset_x = random.randint(-5, 5)
        self.letter_offset_y = random.randint(-5, 5)

        # Create letter Surface
        self.letter_surface = self._create_letter_surface(self.current_letter)

        # Initialize user drawing Surface
        self.user_surface = pygame.Surface((self.width, self.height))
        self.user_surface.fill(self.WHITE)
        self.user_surface.set_colorkey(self.WHITE)

        # Reset state
        self.drawing = False
        self.last_pos = None
        self.show_score = False
        self.score = 0.0
        self.score_timer = 0
        self.start_time = pygame.time.get_ticks()

    def _create_letter_surface(self, letter):
        """Create letter Surface using PIL"""
        # Draw letter with PIL
        pil_size = (self.width, self.height)
        pil_image = Image.new("RGB", pil_size, (255, 255, 255))
        draw = ImageDraw.Draw(pil_image)

        # Get font (system font)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", self.font_size
            )
        except:
            font = ImageFont.load_default()

        # Get bounding box of the letter
        bbox = draw.textbbox((0, 0), letter, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Center placement (with offset)
        x = (self.width - text_width) // 2 + self.letter_offset_x
        y = (self.height - text_height) // 2 + self.letter_offset_y

        # Draw letter in gray
        draw.text((x, y), letter, fill=(200, 200, 200), font=font)

        # Convert PIL image to Pygame Surface
        mode = pil_image.mode
        size = pil_image.size
        data = pil_image.tobytes()

        py_image = pygame.image.fromstring(data, size, mode)

        return py_image

    def calculate_iou(self):
        """Calculate Intersection over Union"""
        # Convert Surface to numpy array
        letter_array = pygame.surfarray.array3d(self.letter_surface)
        user_array = pygame.surfarray.array3d(self.user_surface)

        # Grayscale conversion (simplified: non-black pixels become 1)
        # Letter Surface: 0 for non-gray(200,200,200) pixels
        letter_mask = np.any(letter_array != [255, 255, 255], axis=2).astype(np.float32)

        # User Surface: 1 for non-white(255,255,255) pixels
        user_mask = np.any(user_array != [255, 255, 255], axis=2).astype(np.float32)

        # Intersection and Union
        intersection = np.sum(letter_mask * user_mask)
        union = np.sum(np.maximum(letter_mask, user_mask))

        if union == 0:
            return 0.0

        iou = intersection / union
        return iou

    def handle_events(self):
        """Event processing"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            # Disable mouse operation while showing score
            if self.show_score:
                continue

            # Mouse button pressed
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    # Done button click detection
                    if self.show_done_button and self.button_rect.collidepoint(event.pos):
                        self._on_done_clicked()
                    else:
                        self.drawing = True
                        self.last_pos = event.pos
                        # Draw circle at initial click position
                        pygame.draw.circle(self.user_surface, self.BLACK, event.pos, 12)

            # Mouse button released
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.drawing = False
                    self.last_pos = None

            # Mouse movement
            elif event.type == pygame.MOUSEMOTION:
                if self.drawing:
                    current_pos = event.pos
                    # Draw circle
                    pygame.draw.circle(self.user_surface, self.BLACK, current_pos, 12)
                    self.last_pos = current_pos

        return True

    def _on_done_clicked(self):
        """Process when Done button is clicked"""
        # Calculate IoU score
        self.score = self.calculate_iou()

        # Transition to score display mode
        self.show_score = True
        self.score_timer = pygame.time.get_ticks()

    def update(self):
        """Update game state"""
        # While showing score
        if self.show_score:
            # Move to next letter after 1 second
            if pygame.time.get_ticks() - self.score_timer > 1000:
                self.new_letter()
        else:
            # Time limit check
            if pygame.time.get_ticks() - self.start_time > self.time_limit:
                self._on_done_clicked()

    def draw(self):
        """Draw screen"""
        # Fill background with white
        self.screen.fill(self.WHITE)

        if self.show_score:
            # Display score
            self._draw_score()
        else:
            # Draw letter
            self.screen.blit(self.letter_surface, (0, 0))

            # Display user's drawing
            self.screen.blit(self.user_surface, (0, 0))

            # Draw Done button
            if self.show_done_button:
                self._draw_button()

        pygame.display.flip()

    def _draw_button(self):
        """Draw Done button"""
        # Mouse hover detection
        mouse_pos = pygame.mouse.get_pos()
        color = self.BUTTON_HOVER if self.button_rect.collidepoint(mouse_pos) else self.BUTTON_COLOR

        pygame.draw.rect(self.screen, color, self.button_rect, border_radius=2)

        # Text
        font = pygame.font.Font(None, 16)
        text = font.render("Done", True, self.WHITE)
        text_rect = text.get_rect(center=self.button_rect.center)
        self.screen.blit(text, text_rect)

    def _draw_score(self):
        """Display score at screen center (format easy for OCR detection)"""
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

    sequential = args.mode == "sequential"
    game = LetterTracingGame(sequential, args.show_done_button)
    game.run()
