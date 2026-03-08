# SPDX-License-Identifier: MIT
"""
Generic Gymnasium environment wrapper for GUI + mouse operations

Uses PyAutoGui to operate the actual mouse and capture screenshots.
This allows support for any GUI application (Pygame, browser, desktop apps, etc.).
"""

import re
import subprocess
import time

import cv2
import gymnasium as gym
import numpy as np
import pyautogui
import pytesseract
from gymnasium import spaces
from PIL import Image


def _find_window_region(window_title):
    """Search for window with wmctrl and get client area with xwininfo"""
    # Search window with wmctrl (partial match)
    result = subprocess.run(["wmctrl", "-lG"], capture_output=True, text=True)
    if result.returncode != 0:
        return None

    window_id_hex = None
    matched_title = None

    for line in result.stdout.splitlines():
        # id desk x y w h host title...
        parts = line.split(None, 6)
        if len(parts) < 7:
            continue
        wid_hex, _, _, _, _, _, title = parts
        if window_title.lower() in title.lower():
            window_id_hex = wid_hex
            matched_title = title
            break

    if window_id_hex is None:
        return None

    # Get exact position and size of client area with xwininfo
    xwinfo = subprocess.run(
        ["xwininfo", "-id", window_id_hex],
        capture_output=True,
        text=True,
    )

    if xwinfo.returncode != 0:
        return None

    # Parse xwininfo output
    abs_x = None
    abs_y = None
    width = None
    height = None

    for line in xwinfo.stdout.splitlines():
        line = line.strip()
        if "Absolute upper-left X:" in line:
            abs_x = int(line.split(":")[-1].strip())
        elif "Absolute upper-left Y:" in line:
            abs_y = int(line.split(":")[-1].strip())
        elif "Width:" in line:
            width = int(line.split(":")[-1].strip())
        elif "Height:" in line:
            height = int(line.split(":")[-1].strip())

    if None in [abs_x, abs_y, width, height]:
        return None

    # xwininfo Absolute coordinates point to client area position
    return (abs_x, abs_y, width, height, matched_title)


def activate_window(window_title):
    """Activate window with xdotool"""
    search = subprocess.run(
        ["xdotool", "search", "--onlyvisible", "--limit", "1", "--name", window_title],
        capture_output=True,
        text=True,
    )
    if search.returncode == 0 and search.stdout.strip():
        window_id = search.stdout.splitlines()[0].strip()
        subprocess.run(["xdotool", "windowactivate", "--sync", window_id])
        subprocess.run(["xdotool", "windowraise", window_id])
        return


class GenericGUIEnv(gym.Env):
    """
    Generic GUI environment wrapper (using PyAutoGui)

    Action Space: Box(3,)
        - action[0]: x delta (-1.0 ~ 1.0, relative movement from current position, ±1.0 = half screen width)
        - action[1]: y delta (-1.0 ~ 1.0, relative movement from current position, ±1.0 = half screen height)
        - action[2]: mouse button state (0.0 ~ 1.0, 1 for down, 0 for up)

    Observation Space: Box(height, width, 3)
        - RGB image (uint8)

    Reward:
        - Detected by reward_detector function
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # Observation resize size (height, width) or None
    resize_shape = None

    def __init__(self, reward_detector, render_mode, window_title):
        """
        Args:
            reward_detector: Reward detection function
                - Arguments: (prev_screen: np.ndarray, current_screen: np.ndarray)
                - Returns: float (reward value)
            render_mode: "human" or "rgb_array" or None
            window_title: Window title (partial match) required

        Note:
            - This environment operates mouse at OS level
            - Do not manually move mouse while environment is running
            - PyAutoGui safety feature: Moving mouse to top-left corner triggers emergency stop
        """
        super().__init__()

        self.reward_detector = reward_detector
        self.render_mode = render_mode

        # PyAutoGui settings
        pyautogui.FAILSAFE = True  # Emergency stop at top-left corner
        pyautogui.PAUSE = 0.01  # Wait time between commands (seconds)

        if window_title is None:
            raise ValueError("window_title is required")

        region = _find_window_region(window_title)
        if region is None:
            # Get and display list of available windows
            wmctrl_result = subprocess.run(
                ["wmctrl", "-lG"],
                capture_output=True,
                text=True,
            )
            window_list = []
            if wmctrl_result.returncode == 0:
                for line in wmctrl_result.stdout.splitlines()[:10]:  # Up to 10
                    parts = line.split(None, 6)
                    if len(parts) >= 7:
                        window_list.append(f"  - {parts[6]}")

            available = "\n".join(window_list) if window_list else "  (could not retrieve)"
            raise ValueError(
                f"Window not found: '{window_title}'\n"
                f"Searched with wmctrl but not found.\n"
                f"Available windows (up to 10):\n{available}"
            )

        self.region = tuple(region[:4])
        self.width = region[2]
        self.height = region[3]
        print(f"Window detected: '{region[4]}' at {self.region}")

        # Action space: [dx, dy, button_state]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0]),
            high=np.array([+1.0, +1.0, 1.0]),
            dtype=np.float32,
        )

        # Observation space: RGB image
        if self.resize_shape is not None:
            obs_height, obs_width = self.resize_shape
        else:
            obs_height, obs_width = self.height, self.width

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(obs_height, obs_width, 3), dtype=np.uint8
        )

        # Previous screen
        self.prev_screen = None

        # Step counter (split episodes at arbitrary steps)
        self.step_count = 0

        # Mouse button state
        self.mouse_button_down = False

    def reset(self, seed=None, options=None):
        """
        Reset environment

        Note: This environment does not reset application state.
              Manually reset the application if needed.
        """
        super().reset(seed=seed)

        # Reset previous screen
        self.prev_screen = None

        # Reset step counter
        self.step_count = 0

        # Reset mouse button state
        self.mouse_button_down = False
        pyautogui.mouseUp(button="left")

        # Get initial observation
        observation = self._get_observation()
        self.prev_screen = observation.copy()

        info = {}
        return observation, info

    def step(self, action):
        """
        Execute one step

        Args:
            action: [dx, dy, button_state]
        """
        # Interpret action
        dx_norm, dy_norm, button_state = action

        # Clip delta to -1.0 ~ 1.0 range
        dx_norm_clipped = np.clip(dx_norm, -1.0, 1.0)
        dy_norm_clipped = np.clip(dy_norm, -1.0, 1.0)

        # Convert delta to pixel offset (±1.0 = full screen width/height)
        dx_pixels = int(dx_norm_clipped * (self.width))
        dy_pixels = int(dy_norm_clipped * (self.height))

        # Get current mouse position and compute new position
        current_x, current_y = pyautogui.position()
        region_x, region_y, _, _ = self.region
        new_x = current_x + dx_pixels
        new_y = current_y + dy_pixels

        # Clamp to window region
        new_x = int(np.clip(new_x, region_x, region_x + self.width - 1))
        new_y = int(np.clip(new_y, region_y, region_y + self.height - 1))

        # Move mouse
        pyautogui.moveTo(new_x, new_y)

        # Update mouse button state
        desired_state = button_state > 0.5
        if desired_state and not self.mouse_button_down:
            pyautogui.mouseDown(button="left")
            self.mouse_button_down = True
        elif not desired_state and self.mouse_button_down:
            pyautogui.mouseUp(button="left")
            self.mouse_button_down = False

        # Wait briefly (for application to process)
        time.sleep(0.032)

        # Get observation
        current_screen = self._get_observation()

        # Calculate reward
        reward = 0.0
        if self.reward_detector is not None and self.prev_screen is not None:
            reward = self.reward_detector(self.prev_screen, current_screen)

        # Update previous screen
        self.prev_screen = current_screen.copy()

        # Increment step counter
        self.step_count += 1

        # Check termination
        terminated = False
        truncated = self.step_count >= 200

        info = {}

        return current_screen, reward, terminated, truncated, info

    def _get_observation(self):
        """Get observation (RGB image of screen)"""
        # Capture screenshot
        screenshot = pyautogui.screenshot(region=self.region)

        # Convert to numpy array
        screen_array = np.array(screenshot)

        # Draw mouse cursor on observation
        mx, my = pyautogui.position()
        region_x, region_y, region_w, region_h = self.region
        cx = mx - region_x
        cy = my - region_y
        if 0 <= cx < region_w and 0 <= cy < region_h:
            cv2.circle(screen_array, (cx, cy), 5, (255, 0, 0), -1)
            cv2.circle(screen_array, (cx, cy), 5, (0, 0, 0), 1)

        # Resize processing
        if self.resize_shape is not None:
            height, width = self.resize_shape
            screen_array = cv2.resize(screen_array, (width, height), interpolation=cv2.INTER_AREA)

        return screen_array

    def render(self):
        """Rendering"""
        if self.render_mode == "rgb_array":
            return self.prev_screen
        elif self.render_mode == "human":
            # No rendering needed as actual screen is displayed
            pass

    def close(self):
        """Close environment"""
        # Release mouse button (no problem if already released)
        pyautogui.mouseUp(button="left")


def create_score_reward_detector():
    """
    Generic function to detect reward from score display using OCR

    Can be used for both Letter Tracing Game and Four Quadrant Game

    Returns:
        reward_detector function
    """

    def reward_detector(prev_screen, current_screen):
        """
        Detect score from screen using OCR

        Args:
            prev_screen: Previous screen (H, W, 3)
            current_screen: Current screen (H, W, 3)

        Returns:
            reward: float
        """
        if np.all(current_screen == prev_screen):
            return 0.0
        try:
            # Detect score display area based on color
            lower_bound = (240, 240, 180)  # Light yellow lower bound
            upper_bound = (255, 255, 220)  # Light yellow upper bound
            mask = np.all(
                (current_screen >= lower_bound) & (current_screen <= upper_bound),
                axis=2,
            )

            # Extract region from mask
            if not np.any(mask):
                return 0.0

            # Get mask bounds
            rows, cols = np.where(mask)
            if len(rows) == 0:
                return 0.0

            y1, y2 = rows.min(), rows.max()
            x1, x2 = cols.min(), cols.max()

            # Expand region slightly
            margin = 10
            y1 = max(0, y1 - margin)
            y2 = min(current_screen.shape[0], y2 + margin)
            x1 = max(0, x1 - margin)
            x2 = min(current_screen.shape[1], x2 + margin)

            # Crop score region
            score_region = current_screen[y1:y2, x1:x2, :]

            # Convert to PIL Image
            pil_image = Image.fromarray(score_region)

            # Read text with OCR
            text = pytesseract.image_to_string(pil_image, config="--psm 6").strip()

            # Check if "Score:" keyword is included
            score_keyword = "Score:"
            if score_keyword.lower() not in text.lower():
                return 0.0

            # Extract number (supports negative numbers)
            match = re.search(r"(-?\d+\.\d+)", text)
            if match:
                score = float(match.group(1))
                return score
            else:
                return 0.0

        except Exception as e:
            # Return 0 on OCR failure
            print(f"OCR error: {e}")
            return 0.0

    return reward_detector
