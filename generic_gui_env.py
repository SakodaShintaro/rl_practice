"""
汎用的なGUI + マウス操作のGymnasium環境ラッパー

PyAutoGuiを使って実際のマウスを操作し、スクリーンショットで画面を取得します。
これにより、あらゆるGUIアプリケーション（Pygame、ブラウザ、デスクトップアプリなど）に対応できます。
"""

import re
import shutil
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
    """wmctrl/xdotoolでウィンドウ領域を取得 (部分一致)"""
    if shutil.which("wmctrl"):
        result = subprocess.run(["wmctrl", "-lG"], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                # id desk x y w h host/title...
                parts = line.split(None, 6)
                if len(parts) < 7:
                    continue
                _, _, x, y, w, h, title = parts
                if window_title.lower() in title.lower():
                    return (int(x), int(y), int(w), int(h), title)

    if shutil.which("xdotool"):
        search = subprocess.run(
            ["xdotool", "search", "--name", window_title], capture_output=True, text=True
        )
        if search.returncode == 0 and search.stdout.strip():
            window_id = search.stdout.splitlines()[0].strip()
            geometry = subprocess.run(
                ["xdotool", "getwindowgeometry", "--shell", window_id],
                capture_output=True,
                text=True,
            )
            if geometry.returncode == 0:
                values = {}
                for line in geometry.stdout.splitlines():
                    if "=" in line:
                        key, value = line.split("=", 1)
                        values[key] = value
                required_keys = ("X", "Y", "WIDTH", "HEIGHT")
                if all(key in values for key in required_keys):
                    return (
                        int(values["X"]),
                        int(values["Y"]),
                        int(values["WIDTH"]),
                        int(values["HEIGHT"]),
                        window_title,
                    )

    return None


def activate_window(window_title):
    """xdotool/wmctrlでウィンドウをアクティブ化"""
    if shutil.which("xdotool"):
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

    if shutil.which("wmctrl"):
        subprocess.run(["wmctrl", "-a", window_title])


class GenericGUIEnv(gym.Env):
    """
    汎用的なGUI環境ラッパー（PyAutoGui使用）

    Action Space: Box(4,)
        - action[0]: x座標 (0.0 ~ 1.0, 画面幅に対する相対位置)
        - action[1]: y座標 (0.0 ~ 1.0, 画面高さに対する相対位置)
        - action[2]: KeyDown確率 (0.0 ~ 1.0)
        - action[3]: KeyUp確率 (0.0 ~ 1.0)

    Observation Space: Box(height, width, 3)
        - RGB画像 (uint8)

    Reward:
        - reward_detector関数で検出
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # 観測のリサイズサイズ (height, width) or None
    # サブクラスでオーバーライド可能
    resize_shape = (96, 96)

    def __init__(self, reward_detector, render_mode, window_title):
        """
        Args:
            reward_detector: 報酬検出関数
                - 引数: (prev_screen: np.ndarray, current_screen: np.ndarray)
                - 戻り値: float（報酬値）
            render_mode: "human" or "rgb_array" or None
            window_title: ウィンドウのタイトル（部分一致）必須

        注意:
            - この環境はOSレベルでマウスを操作します
            - 環境実行中は手動でマウスを動かさないでください
            - PyAutoGuiの安全機能: マウスを画面左上隅に移動させると緊急停止します
        """
        super().__init__()

        self.reward_detector = reward_detector
        self.render_mode = render_mode

        # PyAutoGuiの設定
        pyautogui.FAILSAFE = True  # 左上隅で緊急停止
        pyautogui.PAUSE = 0.01  # コマンド間の待機時間（秒）

        if window_title is None:
            raise ValueError("window_titleは必須です")

        region = _find_window_region(window_title)
        if region is None:
            raise ValueError(f"ウィンドウが見つかりません: '{window_title}' (wmctrl/xdotoolで検索)")

        self.region = tuple(region[:4])
        self.width = region[2]
        self.height = region[3]
        print(f"ウィンドウを検出: '{region[4]}' at {self.region}")

        # Action space: [x, y, key_down_prob, key_up_prob]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
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

        # 前回の画面（報酬検出用）
        self.prev_screen = None

    def reset(self, seed=None, options=None):
        """
        環境をリセット

        注意: この環境はアプリケーションの状態をリセットしません。
              必要に応じて、手動でアプリケーションをリセットしてください。
        """
        super().reset(seed=seed)

        # 前回画面をリセット
        self.prev_screen = None

        # 初期観測を取得
        observation = self._get_observation()
        self.prev_screen = observation.copy()

        info = {}
        return observation, info

    def step(self, action):
        """
        1ステップ実行

        Args:
            action: [x, y, key_down_prob, key_up_prob]
        """
        # アクションを解釈
        x_norm, y_norm, key_down_prob, key_up_prob = action

        # 画面座標に変換（region内の相対座標）
        region_x, region_y, _, _ = self.region
        x = int(np.clip(region_x + x_norm * self.width, region_x, region_x + self.width - 1))
        y = int(np.clip(region_y + y_norm * self.height, region_y, region_y + self.height - 1))

        # マウスを移動
        pyautogui.moveTo(x, y)

        # KeyUp/KeyDownの処理（確率的）
        # 同時実行の場合はup->downの順
        if np.random.random() < key_up_prob:
            pyautogui.mouseUp(button="left")

        if np.random.random() < key_down_prob:
            pyautogui.mouseDown(button="left")

        # 少し待機（アプリケーションの処理を待つ）
        time.sleep(0.016)  # 約60fps

        # 観測を取得
        current_screen = self._get_observation()

        # 報酬の計算
        reward = 0.0
        if self.reward_detector is not None and self.prev_screen is not None:
            reward = self.reward_detector(self.prev_screen, current_screen)

        # 前回画面を更新
        self.prev_screen = current_screen.copy()

        # 終了判定（基本的に終了しない）
        terminated = False
        truncated = False

        info = {}

        return current_screen, reward, terminated, truncated, info

    def _get_observation(self):
        """観測（画面のRGB画像）を取得"""
        # スクリーンショットを取得
        screenshot = pyautogui.screenshot(region=self.region)

        # numpy配列に変換
        screen_array = np.array(screenshot)

        # リサイズ処理
        if self.resize_shape is not None:
            height, width = self.resize_shape
            screen_array = cv2.resize(screen_array, (width, height), interpolation=cv2.INTER_AREA)

        return screen_array

    def render(self):
        """レンダリング"""
        if self.render_mode == "rgb_array":
            return self._get_observation()
        elif self.render_mode == "human":
            # 実際の画面が表示されているのでレンダリング不要
            pass

    def close(self):
        """環境をクローズ"""
        # マウスボタンを離す（既に離されていても問題ない）
        pyautogui.mouseUp(button="left")


def create_letter_tracing_reward_detector():
    """
    Letter Tracing Game用の報酬検出関数を生成

    Returns:
        reward_detector関数
    """
    # Letter Tracing Game固有の設定
    score_keyword = "Score:"
    lower_bound = (240, 240, 180)  # 薄い黄色の下限
    upper_bound = (255, 255, 220)  # 薄い黄色の上限

    def reward_detector(prev_screen, current_screen):
        """
        画面からOCRでスコアを検出

        Args:
            prev_screen: 前回の画面 (H, W, 3)
            current_screen: 現在の画面 (H, W, 3)

        Returns:
            reward: float
        """
        if np.all(current_screen == prev_screen):
            return 0.0
        try:
            # 色ベースでスコア表示領域を検出
            mask = np.all(
                (current_screen >= lower_bound) & (current_screen <= upper_bound),
                axis=2,
            )

            # マスクから領域を抽出
            if not np.any(mask):
                return 0.0

            # マスクの範囲を取得
            rows, cols = np.where(mask)
            if len(rows) == 0:
                return 0.0

            y1, y2 = rows.min(), rows.max()
            x1, x2 = cols.min(), cols.max()

            # 領域を少し広げる
            margin = 10
            y1 = max(0, y1 - margin)
            y2 = min(current_screen.shape[0], y2 + margin)
            x1 = max(0, x1 - margin)
            x2 = min(current_screen.shape[1], x2 + margin)

            # スコア領域を切り出し
            score_region = current_screen[y1:y2, x1:x2, :]

            # PIL Imageに変換
            pil_image = Image.fromarray(score_region)

            # OCRで文字列を読み取る
            text = pytesseract.image_to_string(pil_image, config="--psm 6").strip()

            # "Score:" キーワードを含むかチェック
            if score_keyword.lower() not in text.lower():
                return 0.0

            # 数字を抽出
            # "Score: 0.45" のような形式を想定
            match = re.search(r"(\d+\.\d+)", text)
            if match:
                score = float(match.group(1))
                return score
            else:
                return 0.0

        except Exception as e:
            # OCR失敗時は0を返す
            print(f"OCRエラー: {e}")
            return 0.0

    return reward_detector
