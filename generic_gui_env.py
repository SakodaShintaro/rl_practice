"""
汎用的なGUI + マウス操作のGymnasium環境ラッパー

PyAutoGuiを使って実際のマウスを操作し、スクリーンショットで画面を取得します。
これにより、あらゆるGUIアプリケーション（Pygame、ブラウザ、デスクトップアプリなど）に対応できます。
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
    """wmctrlでウィンドウを検索し、xwininfoでクライアント領域を取得"""
    # wmctrlでウィンドウを検索（部分一致）
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

    # xwininfoでクライアント領域の正確な位置とサイズを取得
    xwinfo = subprocess.run(
        ["xwininfo", "-id", window_id_hex],
        capture_output=True,
        text=True,
    )

    if xwinfo.returncode != 0:
        return None

    # xwininfoの出力をパース
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

    # xwininfoのAbsolute座標はクライアント領域の位置を指す
    return (abs_x, abs_y, width, height, matched_title)


def activate_window(window_title):
    """xdotoolでウィンドウをアクティブ化"""
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
    汎用的なGUI環境ラッパー（PyAutoGui使用）

    Action Space: Box(3,)
        - action[0]: x座標 (0.0 ~ 1.0, 画面幅に対する相対位置)
        - action[1]: y座標 (0.0 ~ 1.0, 画面高さに対する相対位置)
        - action[2]: マウスボタン状態 (0.0 ~ 1.0, 1でdown、0でup)

    Observation Space: Box(height, width, 3)
        - RGB画像 (uint8)

    Reward:
        - reward_detector関数で検出
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # 観測のリサイズサイズ (height, width) or None
    resize_shape = (384, 384)

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
            # 利用可能なウィンドウのリストを取得して表示
            wmctrl_result = subprocess.run(
                ["wmctrl", "-lG"],
                capture_output=True,
                text=True,
            )
            window_list = []
            if wmctrl_result.returncode == 0:
                for line in wmctrl_result.stdout.splitlines()[:10]:  # 最大10個まで
                    parts = line.split(None, 6)
                    if len(parts) >= 7:
                        window_list.append(f"  - {parts[6]}")

            available = "\n".join(window_list) if window_list else "  (取得できませんでした)"
            raise ValueError(
                f"ウィンドウが見つかりません: '{window_title}'\n"
                f"wmctrlで検索しましたが見つかりませんでした。\n"
                f"利用可能なウィンドウ（最大10個）:\n{available}"
            )

        self.region = tuple(region[:4])
        self.width = region[2]
        self.height = region[3]
        print(f"ウィンドウを検出: '{region[4]}' at {self.region}")

        # Action space: [x, y, button_state]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
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

        # ステップカウンタ（100ステップごとにエピソードを区切る）
        self.step_count = 0

        # マウスボタンの状態
        self.mouse_button_down = False

    def reset(self, seed=None, options=None):
        """
        環境をリセット

        注意: この環境はアプリケーションの状態をリセットしません。
              必要に応じて、手動でアプリケーションをリセットしてください。
        """
        super().reset(seed=seed)

        # 前回画面をリセット
        self.prev_screen = None

        # ステップカウンタをリセット
        self.step_count = 0

        # マウスボタンの状態をリセット
        self.mouse_button_down = False
        pyautogui.mouseUp(button="left")

        # 初期観測を取得
        observation = self._get_observation()
        self.prev_screen = observation.copy()

        info = {}
        return observation, info

    def step(self, action):
        """
        1ステップ実行

        Args:
            action: [x, y, button_state]
        """
        # アクションを解釈
        x_norm, y_norm, button_state = action

        # アクションを0.0～1.0の範囲にクリップ
        x_norm_clipped = np.clip(x_norm, 0.0, 1.0)
        y_norm_clipped = np.clip(y_norm, 0.0, 1.0)

        # 画面座標に変換（region内の相対座標）
        region_x, region_y, _, _ = self.region
        x = int(region_x + x_norm_clipped * (self.width - 1))
        y = int(region_y + y_norm_clipped * (self.height - 1))

        # マウスを移動
        pyautogui.moveTo(x, y)

        # マウスボタンの状態を更新
        desired_state = button_state > 0.5
        if desired_state and not self.mouse_button_down:
            pyautogui.mouseDown(button="left")
            self.mouse_button_down = True
        elif not desired_state and self.mouse_button_down:
            pyautogui.mouseUp(button="left")
            self.mouse_button_down = False

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

        # ステップカウンタをインクリメント
        self.step_count += 1

        # 終了判定
        terminated = False
        truncated = self.step_count >= 100

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
            lower_bound = (240, 240, 180)  # 薄い黄色の下限
            upper_bound = (255, 255, 220)  # 薄い黄色の上限
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
            score_keyword = "Score:"
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
