"""
人間プレイ用スクリプト

実際のマウス操作をキャプチャして環境に送り、報酬を確認できます。
模倣学習用のデータ収集もサポートします。

注意: このスクリプトを実行する前に、対象のGUIゲームを起動してください。
"""

import argparse
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from pynput import mouse

from generic_gui_env import GenericGUIEnv, activate_window, create_score_reward_detector


def parse_args():
    parser = argparse.ArgumentParser(description="人間プレイ用スクリプト")
    parser.add_argument("--window_title", type=str, default="Letter Tracing Game")
    parser.add_argument("--save_dir", type=str, default="results")
    return parser.parse_args()


class HumanPlayRecorder:
    """人間プレイを記録するクラス"""

    def __init__(self, env, save_dir):
        self.env = env
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(save_dir) / datetime_str

        # マウス状態
        self.current_pos = (0, 0)
        self.button_pressed = False

        # 環境の領域
        if env.region is not None:
            self.region_x, self.region_y, self.region_w, self.region_h = env.region
        else:
            self.region_x, self.region_y = 0, 0
            self.region_w, self.region_h = env.width, env.height

        # 統計情報
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

        # マウスリスナーを開始
        self.listener = mouse.Listener(on_move=self.on_move, on_click=self.on_click)
        self.listener.start()

    def on_move(self, x, y):
        """マウス移動イベント"""
        self.current_pos = (x, y)

    def on_click(self, x, y, button, pressed):
        """マウスクリックイベント"""
        if button == mouse.Button.left:
            self.button_pressed = pressed

    def get_action(self):
        """現在のマウス状態からアクションを生成"""
        x, y = self.current_pos

        # 環境の領域内の相対座標に変換
        x_norm = (x - self.region_x) / self.region_w
        y_norm = (y - self.region_y) / self.region_h

        # クリップ（領域外の場合）
        x_norm = np.clip(x_norm, 0.0, 1.0)
        y_norm = np.clip(y_norm, 0.0, 1.0)

        # マウスボタン状態（押されている=1.0、押されていない=0.0）
        button_state = 1.0 if self.button_pressed else 0.0

        action = np.array([x_norm, y_norm, button_state], dtype=np.float32)
        return action

    def run(self):
        """プレイを実行"""
        print("\n=== 人間プレイ開始 ===")
        print("マウスでゲームをプレイしてください")
        print("Ctrl+C で終了します")
        print()

        # 環境をリセット
        obs, info = self.env.reset()

        start_time = time.time()

        try:
            while True:
                # アクションを取得
                action = self.get_action()

                # ステップ実行
                obs, reward, terminated, truncated, info = self.env.step(action)

                # 統計更新
                self.step_count += 1

                # データ保存
                self._save_step(obs, action, reward)

                if reward != 0:
                    self.total_reward += reward
                    self.reward_history.append(reward)
                    print(f"\n[Step {self.step_count}] 報酬獲得: {reward:.4f}")
                    print(f"累積報酬: {self.total_reward:.4f}")
                    if len(self.reward_history) > 0:
                        avg_reward = np.mean(self.reward_history)
                        print(f"直近平均報酬: {avg_reward:.4f}")

                # 10ステップごとに進捗表示
                if self.step_count % 60 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"[{int(elapsed)}秒経過] Step: {self.step_count}, Total Reward: {self.total_reward:.4f}"
                    )

                # 終了判定
                if terminated or truncated:
                    obs, info = self.env.reset()

        except KeyboardInterrupt:
            print("\n\nプレイを終了します")

        finally:
            # マウスリスナーを停止
            self.listener.stop()

            # 統計情報を表示
            self._print_statistics()

    def _print_statistics(self):
        """統計情報を表示"""
        print("\n=== プレイ統計 ===")
        print(f"総ステップ数: {self.step_count}")
        print(f"累積報酬: {self.total_reward:.4f}")
        if len(self.reward_history) > 0:
            print(f"平均報酬: {np.mean(self.reward_history):.4f}")
            print(f"最大報酬: {np.max(self.reward_history):.4f}")
            print(f"最小報酬: {np.min(self.reward_history):.4f}")

    def _save_step(self, obs, action, reward):
        """各ステップを画像,TSVに保存"""
        img_path = self.images_dir / f"{self.step_count:08d}.png"
        Image.fromarray(obs).save(img_path)

        with open(self.tsv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{self.step_count}\t{float(action[0]):.6f}\t{float(action[1]):.6f}\t"
                f"{float(action[2]):.6f}\t{float(reward):.6f}\n"
            )


def main():
    args = parse_args()

    print("=== 人間プレイ用スクリプト ===")
    print(f"対象ウィンドウ: {args.window_title}")
    print("\n注意: 先にゲームを起動してください")
    print("      ウィンドウが見つかるまで待機します...\n")

    # フォーカスを移す
    activate_window(args.window_title)

    # 報酬検出関数を作成
    reward_detector = create_score_reward_detector()

    env = GenericGUIEnv(
        reward_detector=reward_detector,
        render_mode=None,
        window_title=args.window_title,
    )

    # 人間プレイを開始
    recorder = HumanPlayRecorder(env, save_dir=args.save_dir)

    try:
        recorder.run()
    finally:
        # 環境をクローズ
        env.close()

    print("完了")


if __name__ == "__main__":
    main()
