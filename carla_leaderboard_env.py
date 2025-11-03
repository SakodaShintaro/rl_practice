import os
import sys

# Add CARLA PythonAPI to path
sys.path.append(os.path.expanduser("~/CARLA_0.9.16/PythonAPI/carla"))
sys.path.append(os.path.expanduser("~/work/scenario_runner"))
sys.path.append(os.path.expanduser("~/work/leaderboard"))

import time
from typing import Any, Dict, Optional, Tuple

import carla
import cv2
import gymnasium as gym
import numpy as np


class CARLALeaderboardEnv(gym.Env):
    """
    CARLA Leaderboard準拠のGymnasium環境

    - ルート生成とルート追跡
    - Leaderboard準拠の報酬（Route Completion + Infractions）
    - マップ+ルート+車両位置の可視化
    """

    def __init__(self):
        super().__init__()

        # 設定値
        self.image_size = (192, 192)  # (width, height)
        self.max_episode_steps = 1000
        self.render_mode = "rgb_array"

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, self.image_size[1], self.image_size[0]),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # CARLA接続
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world("Town01")

        # 同期モード設定
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)

        # マップとスポーン地点
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()

        # センサーと車両
        self.vehicle = None
        self.camera_sensor = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None

        # センサーデータ
        self.current_image = None
        self.collision_history = []
        self.lane_invasion_history = []

        # ルート情報
        self.route = []  # List[(carla.Transform, RoadOption)]
        self.route_waypoints = []  # List[carla.Waypoint]
        self.current_waypoint_index = 0

        # Leaderboard評価指標
        self.route_completion = 0.0  # 0.0 ~ 1.0
        self.infractions = {
            "collision_pedestrian": 0,
            "collision_vehicle": 0,
            "collision_static": 0,
            "red_light": 0,
            "lane_invasion": 0,
        }

        # エピソード情報
        self.episode_step = 0

    def reset(
        self,
        seed: Optional[int],
        options: Optional[Dict[str, Any]],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # 既存の車両とセンサーを削除
        if self.camera_sensor is not None:
            self.camera_sensor.destroy()
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
        if self.lane_invasion_sensor is not None:
            self.lane_invasion_sensor.destroy()
        if self.vehicle is not None:
            self.vehicle.destroy()

        # ルート生成
        self.route, self.route_waypoints = self._generate_route()

        # 車両をスポーン（ルートの開始地点）
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]

        spawn_transform = self.route[0][0]
        spawn_transform.location.z += 0.5
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_transform)

        # カメラセンサー
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.image_size[0]))
        camera_bp.set_attribute("image_size_y", str(self.image_size[1]))
        camera_bp.set_attribute("fov", "110")
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera_sensor = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.vehicle
        )
        self.camera_sensor.listen(lambda image: self._process_image(image))

        # 衝突センサー
        collision_bp = blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.vehicle
        )
        self.collision_sensor.listen(lambda event: self._on_collision(event))

        # 車線侵入センサー
        lane_invasion_bp = blueprint_library.find("sensor.other.lane_invasion")
        self.lane_invasion_sensor = self.world.spawn_actor(
            lane_invasion_bp, carla.Transform(), attach_to=self.vehicle
        )
        self.lane_invasion_sensor.listen(lambda event: self._on_lane_invasion(event))

        # 初期化
        self.episode_step = 0
        self.current_waypoint_index = 0
        self.route_completion = 0.0
        self.collision_history = []
        self.lane_invasion_history = []
        self.infractions = {k: 0 for k in self.infractions}
        self.current_image = None

        # 最初のフレームを取得
        for _ in range(5):
            self.world.tick()
            if self.current_image is not None:
                break
            time.sleep(0.1)

        if self.current_image is None:
            self.current_image = np.zeros(
                (3, self.image_size[1], self.image_size[0]), dtype=np.float32
            )

        info = {}
        return self.current_image.copy(), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # 行動を適用
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip((action[1] + 1.0) / 2.0, 0.0, 1.0))
        brake = float(np.clip((action[2] + 1.0) / 2.0, 0.0, 1.0))

        brake = 0.0

        control = carla.VehicleControl(
            steer=steer, throttle=throttle, brake=brake, hand_brake=False, manual_gear_shift=False
        )
        self.vehicle.apply_control(control)

        self.world.tick()

        # ルート追跡の更新
        self._update_route_progress()

        # 報酬計算（Leaderboard準拠）
        reward = self._compute_reward()

        # 終了条件
        terminated = len(self.collision_history) > 0 or self.route_completion >= 1.0
        truncated = self.episode_step >= self.max_episode_steps

        self.episode_step += 1

        obs = (
            self.current_image.copy()
            if self.current_image is not None
            else np.zeros((3, self.image_size[1], self.image_size[0]), dtype=np.float32)
        )

        info = {
            "route_completion": self.route_completion,
            "infractions": self.infractions.copy(),
            "speed": self._get_speed(),
            "distance_to_route": self._get_distance_to_route(),
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """マップ+ルート+車両位置を可視化"""
        if self.render_mode != "rgb_array":
            return None

        # マップサイズを計算
        map_size = 512
        render_img = np.ones((map_size, map_size, 3), dtype=np.uint8) * 200

        # 車両の現在位置
        vehicle_loc = self.vehicle.get_location()

        # ルートを描画
        for i in range(len(self.route_waypoints) - 1):
            wp1 = self.route_waypoints[i]
            wp2 = self.route_waypoints[i + 1]

            x1, y1 = self._world_to_pixel(wp1.transform.location, vehicle_loc, map_size)
            x2, y2 = self._world_to_pixel(wp2.transform.location, vehicle_loc, map_size)

            # ルートを青線で描画
            color = (255, 0, 0) if i < self.current_waypoint_index else (100, 100, 255)
            cv2.line(render_img, (x1, y1), (x2, y2), color, 2)

        # 車両位置を描画（赤丸）
        center_x, center_y = map_size // 2, map_size // 2
        cv2.circle(render_img, (center_x, center_y), 8, (0, 0, 255), -1)

        # 車両の向き（矢印）
        vehicle_transform = self.vehicle.get_transform()
        yaw = np.radians(vehicle_transform.rotation.yaw)
        arrow_length = 20
        end_x = int(center_x + arrow_length * np.cos(yaw))
        end_y = int(center_y - arrow_length * np.sin(yaw))
        cv2.arrowedLine(render_img, (center_x, center_y), (end_x, end_y), (0, 255, 0), 2)

        # 情報テキスト
        cv2.putText(
            render_img,
            f"RC: {self.route_completion:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            render_img,
            f"Speed: {self._get_speed():.1f} km/h",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

        return render_img

    def close(self):
        if self.camera_sensor is not None:
            self.camera_sensor.destroy()
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
        if self.lane_invasion_sensor is not None:
            self.lane_invasion_sensor.destroy()
        if self.vehicle is not None:
            self.vehicle.destroy()

        if self.world is not None:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)

    def _generate_route(self) -> Tuple[list, list]:
        """ランダムなルートを生成"""
        start_wp = self.map.get_waypoint(
            np.random.choice(self.spawn_points).location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        route_waypoints = [start_wp]
        current_wp = start_wp
        distance = 0.0

        while distance < 200.0:
            next_wps = current_wp.next(2.0)
            if not next_wps:
                break

            # ランダムに次のwaypointを選択（交差点では分岐）
            current_wp = np.random.choice(next_wps)
            route_waypoints.append(current_wp)

            if len(route_waypoints) > 1:
                prev_loc = route_waypoints[-2].transform.location
                curr_loc = current_wp.transform.location
                distance += prev_loc.distance(curr_loc)

        # Transformのリストに変換
        route = [(wp.transform, None) for wp in route_waypoints]

        return route, route_waypoints

    def _update_route_progress(self):
        """ルート進行状況を更新"""
        vehicle_loc = self.vehicle.get_location()

        # 最も近いwaypointを探す
        min_dist = float("inf")
        closest_idx = self.current_waypoint_index

        for i in range(self.current_waypoint_index, len(self.route_waypoints)):
            wp_loc = self.route_waypoints[i].transform.location
            dist = vehicle_loc.distance(wp_loc)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        self.current_waypoint_index = closest_idx
        self.route_completion = min(1.0, closest_idx / max(1, len(self.route_waypoints) - 1))

    def _compute_reward(self) -> float:
        """
        Leaderboard準拠の報酬計算
        Driving Score = Route Completion x Infraction Penalty
        """
        # Route Completion報酬
        rc_reward = self.route_completion * 10.0

        # Infraction Penalty（Leaderboardのペナルティ値）
        penalty = 0.0
        if len(self.collision_history) > 0:
            penalty += 0.65  # 衝突ペナルティ
        if len(self.lane_invasion_history) > 0:
            penalty += 0.1 * len(self.lane_invasion_history)  # 車線逸脱

        # ルートからの距離ペナルティ
        dist_to_route = self._get_distance_to_route()
        if dist_to_route > 5.0:  # 5m以上離れたらペナルティ
            penalty += 0.5

        # 速度報酬（適度な速度を奨励）
        speed = self._get_speed()
        speed_reward = 0.01 if 10.0 < speed < 50.0 else 0.0

        return rc_reward + speed_reward - penalty

    def _get_distance_to_route(self) -> float:
        """車両とルートの最短距離"""
        if self.current_waypoint_index >= len(self.route_waypoints):
            return 0.0

        vehicle_loc = self.vehicle.get_location()
        wp = self.route_waypoints[self.current_waypoint_index]
        return vehicle_loc.distance(wp.transform.location)

    def _world_to_pixel(
        self, world_loc: carla.Location, center_loc: carla.Location, map_size: int
    ) -> Tuple[int, int]:
        """ワールド座標をピクセル座標に変換"""
        scale = 4.0  # メートル/ピクセル
        x = int((world_loc.x - center_loc.x) / scale + map_size // 2)
        y = int(-(world_loc.y - center_loc.y) / scale + map_size // 2)
        return x, y

    def _process_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]
        array = array.transpose(2, 0, 1).astype(np.float32) / 255.0
        self.current_image = array

    def _on_collision(self, event):
        self.collision_history.append(event)
        other_actor = event.other_actor
        if "vehicle" in other_actor.type_id:
            self.infractions["collision_vehicle"] += 1
        elif "walker" in other_actor.type_id:
            self.infractions["collision_pedestrian"] += 1
        else:
            self.infractions["collision_static"] += 1

    def _on_lane_invasion(self, event):
        self.lane_invasion_history.append(event)
        self.infractions["lane_invasion"] += 1

    def _get_speed(self) -> float:
        velocity = self.vehicle.get_velocity()
        return 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
