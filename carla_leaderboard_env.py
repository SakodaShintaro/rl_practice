import os
import sys

# Add CARLA PythonAPI to path
sys.path.append(os.path.expanduser("~/CARLA_0.9.16/PythonAPI/carla"))
sys.path.append(os.path.expanduser("~/work/scenario_runner"))
sys.path.append(os.path.expanduser("~/work/leaderboard"))

import time
from typing import Any, Dict, Optional, Tuple

import carla
import gymnasium as gym
import numpy as np


class CARLALeaderboardEnv(gym.Env):
    """
    CARLA Leaderboard準拠のGymnasium環境

    観測空間: RGB画像 (C, H, W) 正規化済み
    行動空間: 連続制御 [steer, throttle, brake] それぞれ[-1, 1]
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        host: str,
        port: int,
        town: str,
        image_size: Tuple[int, int],
        max_episode_steps: int,
        render_mode: Optional[str],
    ):
        super().__init__()

        self.host = host
        self.port = port
        self.town = town
        self.image_size = image_size  # (width, height)
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        # Gymnasium spaces
        # 観測: 正規化されたRGB画像 (C, H, W)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, image_size[1], image_size[0]),  # (C, H, W)
            dtype=np.float32,
        )

        # 行動: [steer, throttle, brake]
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # CARLA接続
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(self.town)

        # 同期モード設定
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)

        # スポーン地点取得
        self.spawn_points = self.world.get_map().get_spawn_points()

        # センサーと車両（エピソードごとに再生成）
        self.vehicle = None
        self.camera_sensor = None
        self.collision_sensor = None

        # センサーデータ
        self.current_image = None
        self.collision_history = []

        # エピソード情報
        self.episode_step = 0
        self.total_reward = 0.0
        self.prev_location = None

    def reset(
        self,
        seed: Optional[int],
        options: Optional[Dict[str, Any]],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # 既存の車両とセンサーを削除
        if self.camera_sensor is not None:
            self.camera_sensor.destroy()
            self.camera_sensor = None
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
            self.collision_sensor = None
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None

        # 車両をスポーン
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]

        spawn_point = np.random.choice(self.spawn_points)
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.prev_location = self.vehicle.get_location()

        # カメラセンサーを設置
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.image_size[0]))
        camera_bp.set_attribute("image_size_y", str(self.image_size[1]))
        camera_bp.set_attribute("fov", "110")

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera_sensor = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.vehicle
        )
        self.camera_sensor.listen(lambda image: self._process_image(image))

        # 衝突センサーを設置
        collision_bp = blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.vehicle
        )
        self.collision_sensor.listen(lambda event: self._on_collision(event))

        # 初期化
        self.episode_step = 0
        self.total_reward = 0.0
        self.collision_history = []
        self.current_image = None

        # 最初のフレームを取得
        for _ in range(5):  # センサーデータが来るまで待つ
            self.world.tick()
            if self.current_image is not None:
                break
            time.sleep(0.1)

        if self.current_image is None:
            # フォールバック: ゼロ画像
            self.current_image = np.zeros(
                (3, self.image_size[1], self.image_size[0]), dtype=np.float32
            )

        info = {}
        return self.current_image.copy(), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # 行動を適用
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip((action[1] + 1.0) / 2.0, 0.0, 1.0))  # [-1,1] -> [0,1]
        brake = float(np.clip((action[2] + 1.0) / 2.0, 0.0, 1.0))  # [-1,1] -> [0,1]

        brake = 0.0

        control = carla.VehicleControl(
            steer=steer, throttle=throttle, brake=brake, hand_brake=False, manual_gear_shift=False
        )
        self.vehicle.apply_control(control)

        # シミュレーションを1ステップ進める
        self.world.tick()

        # 報酬計算
        reward = self._compute_reward()

        # 終了条件チェック
        terminated = len(self.collision_history) > 0
        truncated = self.episode_step >= self.max_episode_steps

        self.episode_step += 1
        self.total_reward += reward

        # 観測取得
        if self.current_image is None:
            obs = np.zeros((3, self.image_size[1], self.image_size[0]), dtype=np.float32)
        else:
            obs = self.current_image.copy()

        info = {
            "collision": len(self.collision_history) > 0,
            "speed": self._get_speed(),
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array" and self.current_image is not None:
            # (C, H, W) -> (H, W, C) に変換して [0, 255] にスケール
            img = self.current_image.transpose(1, 2, 0) * 255.0
            return img.astype(np.uint8)
        return None

    def close(self):
        # センサーと車両を削除
        if self.camera_sensor is not None:
            self.camera_sensor.destroy()
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
        if self.vehicle is not None:
            self.vehicle.destroy()

        # 非同期モードに戻す
        if self.world is not None:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)

    def _process_image(self, image):
        """カメラ画像を処理"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))  # BGRA
        array = array[:, :, :3]  # RGB

        # (H, W, C) -> (C, H, W) に変換して正規化
        array = array.transpose(2, 0, 1).astype(np.float32) / 255.0
        self.current_image = array

    def _on_collision(self, event):
        """衝突イベントのコールバック"""
        self.collision_history.append(event)

    def _get_speed(self) -> float:
        """車両の速度を取得 (km/h)"""
        velocity = self.vehicle.get_velocity()
        speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        return speed

    def _compute_reward(self) -> float:
        """
        報酬を計算
        - 前進距離に比例
        - 速度にボーナス
        - 衝突にペナルティ
        """
        reward = 0.0

        # 前進距離
        current_location = self.vehicle.get_location()
        if self.prev_location is not None:
            distance = np.sqrt(
                (current_location.x - self.prev_location.x) ** 2
                + (current_location.y - self.prev_location.y) ** 2
            )
            reward += distance * 0.1  # スケール調整
        self.prev_location = current_location

        # 速度ボーナス (適度な速度を奨励)
        speed = self._get_speed()
        target_speed = 30.0  # km/h
        speed_reward = 1.0 - abs(speed - target_speed) / target_speed
        reward += max(0.0, speed_reward) * 0.01

        # 衝突ペナルティ
        if len(self.collision_history) > 0:
            reward -= 10.0

        return reward
