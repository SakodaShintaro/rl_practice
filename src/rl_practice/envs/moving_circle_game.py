# SPDX-License-Identifier: MIT
"""
Moving Circle Click Game
Three types of circles move around the screen:
- Green: large, slow, reward +0.1
- Yellow: medium, medium speed, reward +0.5
- Red: small, fast, reward +1.0
Each circle moves in a random direction for a random number of steps,
then picks a new random direction.
"""

import argparse
import math
import random

import numpy as np
import pygame

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_green", type=int, default=1)
    parser.add_argument("--num_yellow", type=int, default=1)
    parser.add_argument("--num_red", type=int, default=1)
    return parser.parse_args()


class MovingCircleGame:
    def __init__(self, num_green: int, num_yellow: int, num_red: int) -> None:
        pygame.init()

        self.width = 192
        self.height = 192
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Moving Circle Game")
        self.clock = pygame.time.Clock()

        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.SCORE_BG = (255, 255, 200)
        self.SCORE_BORDER = (200, 150, 0)

        self.state = STATE_PLAYING
        self.score = 0.0
        self.total_score = 0.0
        self.state_timer = 0
        self.score_duration = 250
        self.cooldown_duration = 150
        self.cooldown_end_time = 0

        counts = {"green": num_green, "yellow": num_yellow, "red": num_red}
        self.circles = []
        for name, config in CIRCLE_TYPES.items():
            for _ in range(counts[name]):
                self.circles.append(self._create_circle(config))

    def _create_circle(self, config: dict) -> dict:
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

    def _respawn_circle(self, circle: dict) -> None:
        circle["x"] = random.uniform(circle["radius"], self.width - circle["radius"])
        circle["y"] = random.uniform(circle["radius"], self.height - circle["radius"])
        circle["direction"] = random.uniform(0, 2 * math.pi)
        circle["steps_remaining"] = random.randint(circle["min_steps"], circle["max_steps"])

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        current_time = pygame.time.get_ticks()
        if current_time < self.cooldown_end_time:
            return True

        mouse_pressed = pygame.mouse.get_pressed()[0]
        if self.state == STATE_PLAYING and mouse_pressed:
            self._on_click(pygame.mouse.get_pos())

        return True

    def _on_click(self, pos: tuple[int, int]) -> None:
        clicked_circle = None
        best_reward = -1.0
        for circle in self.circles:
            dx = pos[0] - circle["x"]
            dy = pos[1] - circle["y"]
            if dx * dx + dy * dy <= circle["radius"] ** 2 and circle["reward"] > best_reward:
                clicked_circle = circle
                best_reward = circle["reward"]

        self.score = max(0.0, best_reward)
        self.total_score += self.score
        if clicked_circle is not None:
            self._respawn_circle(clicked_circle)
        self.state = STATE_SHOW_SCORE
        self.state_timer = pygame.time.get_ticks()

    def _move_circles(self) -> None:
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
                circle["steps_remaining"] = random.randint(circle["min_steps"], circle["max_steps"])

    def update(self) -> None:
        self._move_circles()

        if self.state == STATE_SHOW_SCORE:
            if pygame.time.get_ticks() - self.state_timer > self.score_duration:
                self.state = STATE_PLAYING
                self.cooldown_end_time = pygame.time.get_ticks() + self.cooldown_duration

    def draw(self) -> None:
        self.screen.fill(self.WHITE)

        for circle in self.circles:
            pygame.draw.circle(
                self.screen,
                circle["color"],
                (int(circle["x"]), int(circle["y"])),
                circle["radius"],
            )

        if self.state == STATE_SHOW_SCORE:
            self._draw_score()

        self._draw_total_score()
        pygame.display.flip()

    def _draw_score(self) -> None:
        score_box_width = 120
        score_box_height = 60
        score_box_x = (self.width - score_box_width) // 2
        score_box_y = (self.height - score_box_height) // 2
        score_box_rect = pygame.Rect(score_box_x, score_box_y, score_box_width, score_box_height)

        pygame.draw.rect(self.screen, self.SCORE_BG, score_box_rect)
        pygame.draw.rect(self.screen, self.SCORE_BORDER, score_box_rect, 2)

        label_font = pygame.font.Font(None, 24)
        label_text = label_font.render("Score:", True, self.BLACK)
        label_rect = label_text.get_rect(center=(self.width // 2, self.height // 2 - 10))
        self.screen.blit(label_text, label_rect)

        score_font = pygame.font.Font(None, 30)
        score_surface = score_font.render(f"{self.score:+.1f}", True, self.BLACK)
        score_rect = score_surface.get_rect(center=(self.width // 2, self.height // 2 + 12))
        self.screen.blit(score_surface, score_rect)

    def _draw_total_score(self) -> None:
        font = pygame.font.Font(None, 20)
        text = font.render(f"Total: {self.total_score:.1f}", True, self.BLACK)
        self.screen.blit(text, (5, 5))

    def get_screen_array(self) -> np.ndarray:
        array = pygame.surfarray.array3d(self.screen)
        array = np.transpose(array, (1, 0, 2))
        return array

    def run(self) -> None:
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)
        pygame.quit()


if __name__ == "__main__":
    args = parse_args()
    game = MovingCircleGame(args.num_green, args.num_yellow, args.num_red)
    game.run()
