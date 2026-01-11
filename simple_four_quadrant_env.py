"""
Four Quadrant Game - Pygameを使わずnumpyで実装
画面を4分割し、1区画だけ赤色、残りは白色
赤色クリック→報酬+1、白色クリック→報酬-0.1
"""

import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SimpleFourQuadrantEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode):
        super().__init__()

        self.render_mode = render_mode

        # 固定値
        self.width = 192
        self.height = 192

        # 色定義 (RGB)
        self.WHITE = np.array([255, 255, 255], dtype=np.uint8)
        self.BLACK = np.array([0, 0, 0], dtype=np.uint8)
        self.RED = np.array([255, 0, 0], dtype=np.uint8)

        # 4分割の矩形を定義 (x, y, w, h)
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

        # Observation space: RGB画像
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 3),
            dtype=np.uint8,
        )

        # 現在の正解の区画インデックス
        self.correct_quadrant = 0

        # ステップカウンタ
        self.step_count = 0

        # マウスボタンの状態
        self.prev_button_state = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 新しい問題を生成
        self.correct_quadrant = random.randint(0, 3)
        self.step_count = 0
        self.prev_button_state = False

        # 初期観測を取得
        observation = self._get_observation()

        info = {}
        return observation, info

    def step(self, action):
        # ステップカウンタをインクリメント
        self.step_count += 1

        # アクションを解釈
        x_norm, y_norm, button_state = action

        # アクションを0.0～1.0の範囲にクリップ
        x_norm = np.clip(x_norm, 0.0, 1.0)
        y_norm = np.clip(y_norm, 0.0, 1.0)

        # 画面座標に変換
        x = int(x_norm * (self.width - 1))
        y = int(y_norm * (self.height - 1))

        # ボタンの状態を判定
        current_button_state = button_state > 0.5

        # クリック判定（ボタンが押されている間）
        reward = -0.5
        if current_button_state:
            # どの区画がクリックされたか判定
            clicked_quadrant = None
            for i, (qx, qy, qw, qh) in enumerate(self.quadrants):
                if qx <= x < qx + qw and qy <= y < qy + qh:
                    clicked_quadrant = i
                    break

            # 報酬を計算
            if clicked_quadrant == self.correct_quadrant:
                reward = 1.0
            else:
                reward = 0.0

            # 新しい問題を生成
            self.correct_quadrant = random.randint(0, 3)

        # ボタンの状態を更新
        self.prev_button_state = current_button_state

        # 観測を取得
        observation = self._get_observation()

        # 終了判定
        terminated = False
        truncated = self.step_count >= 200

        info = {}

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """観測（画面のRGB画像）を取得"""
        return self._render_frame()

    def _render_frame(self):
        """画面描画 - numpyで直接生成"""
        # 白背景の画像を作成
        image = np.full((self.height, self.width, 3), self.WHITE, dtype=np.uint8)

        # 各区画を描画
        for i, (x, y, w, h) in enumerate(self.quadrants):
            if i == self.correct_quadrant:
                # 正解の区画は赤色で塗りつぶし
                image[y : y + h, x : x + w] = self.RED
            else:
                # それ以外は枠線だけ描く（黒い線）
                image[y, x : x + w] = self.BLACK
                image[y + h - 1, x : x + w] = self.BLACK
                image[y : y + h, x] = self.BLACK
                image[y : y + h, x + w - 1] = self.BLACK

        return image

    def render(self):
        """レンダリング"""
        return self._render_frame()

    def close(self):
        """環境をクローズ"""
        pass
