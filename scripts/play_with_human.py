"""
Human play script

Captures actual mouse operations and sends them to the environment to check rewards.
Also supports data collection for imitation learning.

Note: Please start the target GUI game before running this script.
"""

import argparse
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from generic_gui_env import GenericGUIEnv, activate_window, create_score_reward_detector
from PIL import Image
from pynput import mouse


def parse_args():
    parser = argparse.ArgumentParser(description="Human play script")
    parser.add_argument("--window_title", type=str, default="Letter Tracing Game")
    parser.add_argument("--save_dir", type=str, default="results")
    return parser.parse_args()


class HumanPlayRecorder:
    """Class to record human play"""

    def __init__(self, env, save_dir):
        self.env = env
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(save_dir) / datetime_str

        # Mouse state
        self.current_pos = (0, 0)
        self.button_pressed = False

        # Environment region
        if env.region is not None:
            self.region_x, self.region_y, self.region_w, self.region_h = env.region
        else:
            self.region_x, self.region_y = 0, 0
            self.region_w, self.region_h = env.width, env.height

        # Statistics
        self.step_count = 0
        self.total_reward = 0.0
        self.reward_history = deque(maxlen=10)

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.save_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.tsv_path = self.save_dir / "log.tsv"
        if not self.tsv_path.exists():
            with open(self.tsv_path, "w", encoding="utf-8") as f:
                f.write("step\taction0\taction1\taction2\treward\n")

        # Start mouse listener
        self.listener = mouse.Listener(on_move=self.on_move, on_click=self.on_click)
        self.listener.start()

    def on_move(self, x, y):
        """Mouse move event"""
        self.current_pos = (x, y)

    def on_click(self, x, y, button, pressed):
        """Mouse click event"""
        if button == mouse.Button.left:
            self.button_pressed = pressed

    def get_action(self):
        """Generate action from current mouse state"""
        x, y = self.current_pos

        # Convert to relative coordinates within environment region
        x_norm = (x - self.region_x) / self.region_w
        y_norm = (y - self.region_y) / self.region_h

        # Clip (when outside region)
        x_norm = np.clip(x_norm, 0.0, 1.0)
        y_norm = np.clip(y_norm, 0.0, 1.0)

        # Mouse button state (pressed=1.0, not pressed=0.0)
        button_state = 1.0 if self.button_pressed else 0.0

        action = np.array([x_norm, y_norm, button_state], dtype=np.float32)
        return action

    def run(self):
        """Execute play"""
        print("\n=== Human Play Started ===")
        print("Play the game with mouse")
        print("Press Ctrl+C to exit")
        print()

        # Reset environment
        obs, info = self.env.reset()

        start_time = time.time()

        try:
            while True:
                # Get action
                action = self.get_action()

                # Execute step
                obs, reward, terminated, truncated, info = self.env.step(action)

                # Update statistics
                self.step_count += 1

                # Save data
                self._save_step(obs, action, reward)

                if reward != 0:
                    self.total_reward += reward
                    self.reward_history.append(reward)
                    print(f"\n[Step {self.step_count}] Reward obtained: {reward:.4f}")
                    print(f"Cumulative reward: {self.total_reward:.4f}")
                    if len(self.reward_history) > 0:
                        avg_reward = np.mean(self.reward_history)
                        print(f"Recent average reward: {avg_reward:.4f}")

                # Show progress every 60 steps
                if self.step_count % 60 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"[{int(elapsed)} sec elapsed] Step: {self.step_count}, Total Reward: {self.total_reward:.4f}"
                    )

                # Check termination
                if terminated or truncated:
                    obs, info = self.env.reset()

        except KeyboardInterrupt:
            print("\n\nEnding play")

        finally:
            # Stop mouse listener
            self.listener.stop()

            # Display statistics
            self._print_statistics()

    def _print_statistics(self):
        """Display statistics"""
        print("\n=== Play Statistics ===")
        print(f"Total steps: {self.step_count}")
        print(f"Cumulative reward: {self.total_reward:.4f}")
        if len(self.reward_history) > 0:
            print(f"Average reward: {np.mean(self.reward_history):.4f}")
            print(f"Max reward: {np.max(self.reward_history):.4f}")
            print(f"Min reward: {np.min(self.reward_history):.4f}")

    def _save_step(self, obs, action, reward):
        """Save each step to image and TSV"""
        img_path = self.images_dir / f"{self.step_count:08d}.png"
        Image.fromarray(obs).save(img_path)

        with open(self.tsv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{self.step_count}\t{float(action[0]):.6f}\t{float(action[1]):.6f}\t"
                f"{float(action[2]):.6f}\t{float(reward):.6f}\n"
            )


def main():
    args = parse_args()

    print("=== Human Play Script ===")
    print(f"Target window: {args.window_title}")
    print("\nNote: Please start the game first")
    print("      Waiting until window is found...\n")

    # Focus the window
    activate_window(args.window_title)

    # Create reward detector function
    reward_detector = create_score_reward_detector()

    env = GenericGUIEnv(
        reward_detector=reward_detector,
        render_mode=None,
        window_title=args.window_title,
    )

    # Start human play
    recorder = HumanPlayRecorder(env, save_dir=args.save_dir)

    try:
        recorder.run()
    finally:
        # Close environment
        env.close()

    print("Done")


if __name__ == "__main__":
    main()
