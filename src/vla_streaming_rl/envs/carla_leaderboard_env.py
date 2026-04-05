# SPDX-License-Identifier: MIT
import copy
import time
from dataclasses import dataclass

import carla
import cv2
import gymnasium as gym
import numpy as np
import scipy

from vla_streaming_rl.envs.vehicle_graph import overlay_vehicle_graphs

# Comfort thresholds (from Alpamayo comfort_reward.py)
# https://github.com/NVlabs/alpamayo/blob/main/finetune/rl/rewards/comfort_reward.py
COMFORT_MAX_ABS_MAG_JERK = 8.37  # [m/s^3]
COMFORT_MAX_ABS_LAT_ACCEL = 4.89  # [m/s^2]
COMFORT_MAX_LON_ACCEL = 2.40  # [m/s^2]
COMFORT_MIN_LON_ACCEL = -4.05  # [m/s^2]
COMFORT_MAX_ABS_LON_JERK = 4.13  # [m/s^3]
COMFORT_MAX_ABS_YAW_RATE = 0.95  # [rad/s]
COMFORT_MAX_ABS_YAW_ACCEL = 1.93  # [rad/s^2]

DT = 0.1  # [s] (10 FPS)


@dataclass
class VehiclePhysics:
    """Vehicle physics and action state for a single timestep."""

    acceleration_vector: np.ndarray
    angular_velocity_vector: np.ndarray
    velocity: np.ndarray
    velocity_kph: float
    acceleration: float
    lon_acceleration: float
    lat_acceleration: float
    jerk: float
    lon_jerk: float
    prev_lon_acceleration: float
    angular_velocity: float
    angular_acceleration: float
    throttle: float
    brake: float
    steering: float

    def update(self, vehicle, throttle: float, brake: float, steering: float):
        """Update vehicle physics information and action state."""
        self.throttle = throttle
        self.brake = brake
        self.steering = steering

        # Calculate velocity
        vel = vehicle.get_velocity()
        self.velocity = np.array([vel.x, vel.y, vel.z])
        self.velocity_kph = 3.6 * np.linalg.norm(self.velocity)

        # Vehicle forward/right vectors from yaw
        yaw = np.radians(vehicle.get_transform().rotation.yaw)
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])

        # Calculate acceleration
        acceleration = vehicle.get_acceleration()
        acceleration_vec = np.array([acceleration.x, acceleration.y, acceleration.z])
        self.acceleration = np.linalg.norm(acceleration_vec)
        self.lon_acceleration = float(np.dot(acceleration_vec, forward))
        self.lat_acceleration = float(np.dot(acceleration_vec, right))

        # Calculate jerk
        jerk_vec = (acceleration_vec - self.acceleration_vector) / DT
        self.jerk = np.linalg.norm(jerk_vec)
        self.lon_jerk = (self.lon_acceleration - self.prev_lon_acceleration) / DT
        self.prev_lon_acceleration = self.lon_acceleration
        self.acceleration_vector = acceleration_vec

        # Calculate angular velocity
        angular_velocity = vehicle.get_angular_velocity()
        angular_velocity_vec = np.array(
            [angular_velocity.x, angular_velocity.y, angular_velocity.z]
        )
        self.angular_velocity = np.linalg.norm(angular_velocity_vec)

        # Calculate angular acceleration
        angular_accel_vec = (angular_velocity_vec - self.angular_velocity_vector) / DT
        self.angular_acceleration = np.linalg.norm(angular_accel_vec)
        self.angular_velocity_vector = angular_velocity_vec

    @classmethod
    def create(cls) -> "VehiclePhysics":
        return cls(
            acceleration_vector=np.zeros(3),
            angular_velocity_vector=np.zeros(3),
            velocity=np.zeros(3),
            velocity_kph=0.0,
            acceleration=0.0,
            lon_acceleration=0.0,
            lat_acceleration=0.0,
            jerk=0.0,
            lon_jerk=0.0,
            prev_lon_acceleration=0.0,
            angular_velocity=0.0,
            angular_acceleration=0.0,
            throttle=0.0,
            brake=0.0,
            steering=0.0,
        )


class CARLALeaderboardEnv(gym.Env):
    """
    CARLA Leaderboard-compliant Gymnasium environment

    - Route generation and route tracking
    - Leaderboard-compliant reward (Route Completion + Infractions)
    - Visualization of map + route + vehicle position
    """

    def __init__(self):
        super().__init__()
        self.prompt = "Drive a car along a route in CARLA. Follow the planned route, obey traffic rules, and avoid collisions."

        # Configuration values
        self.image_size = (256, 256)  # (width, height)
        self.max_episode_steps = 1000
        self.render_mode = "rgb_array"
        self.fps = 10  # Simulation FPS

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, self.image_size[1], self.image_size[0]),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # CARLA connection
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world("Town01")

        # Synchronous mode settings
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        self.world.apply_settings(settings)

        # Map and spawn points
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()

        # Sensors and vehicle
        self.vehicle = None
        self.camera_sensor = None
        self.third_person_camera = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None

        # Sensor data
        self.current_image = None
        self.third_person_image = None
        self.lane_invasion_history = []

        # Route information
        self.route_waypoints = []  # List[carla.Waypoint]
        self.current_waypoint_index = 0

        # Leaderboard evaluation metrics
        self.route_completion = 0.0  # 0.0 ~ 1.0
        self.infractions = {
            "collision_pedestrian": 0,
            "collision_vehicle": 0,
            "collision_static": 0,
            "red_light": 0,
            "lane_invasion": 0,
        }
        self.prev_driving_score = 0.0

        # Episode information
        self.episode_step = 0
        self.negative_reward_count = 0  # Count of consecutive negative rewards
        self.min_distance_to_route = 0.0  # Minimum distance to route

        # Vehicle physics information
        self.vehicle_physics = VehiclePhysics.create()
        self.current_action = np.zeros(2, dtype=np.float32)

        # History buffer for render graphs
        self.history_length = 200
        self.physics_history: list[VehiclePhysics] = []

    def reset(
        self,
        seed: int | None,
        options: dict[str, any] | None,
    ) -> tuple[np.ndarray, dict[str, any]]:
        super().reset(seed=seed)

        # Delete existing vehicle and sensors
        if self.camera_sensor is not None:
            self.camera_sensor.destroy()
        if self.third_person_camera is not None:
            self.third_person_camera.destroy()
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
        if self.lane_invasion_sensor is not None:
            self.lane_invasion_sensor.destroy()
        if self.vehicle is not None:
            self.vehicle.destroy()

        # Generate route
        self.route_waypoints, start_pose = self._generate_route()

        # Spawn vehicle (at route start point)
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]

        while True:
            try:
                spawn_transform = start_pose
                spawn_transform.location.z += 0.5
                self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_transform)
                break
            except RuntimeError:
                time.sleep(0.1)
                continue

        # Camera sensor
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.image_size[0]))
        camera_bp.set_attribute("image_size_y", str(self.image_size[1]))
        camera_bp.set_attribute("fov", "110")
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera_sensor = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.vehicle
        )
        self.camera_sensor.listen(lambda image: self._process_image(image))

        # Third-person camera (for human monitoring)
        tp_bp = blueprint_library.find("sensor.camera.rgb")
        tp_bp.set_attribute("image_size_x", "800")
        tp_bp.set_attribute("image_size_y", "600")
        tp_bp.set_attribute("fov", "90")
        tp_transform = carla.Transform(carla.Location(x=-8.0, z=5.0), carla.Rotation(pitch=-20.0))
        self.third_person_camera = self.world.spawn_actor(
            tp_bp, tp_transform, attach_to=self.vehicle
        )
        self.third_person_camera.listen(lambda image: self._process_third_person_image(image))

        # Collision sensor
        collision_bp = blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.vehicle
        )
        self.collision_sensor.listen(lambda event: self._on_collision(event))

        # Lane invasion sensor
        lane_invasion_bp = blueprint_library.find("sensor.other.lane_invasion")
        self.lane_invasion_sensor = self.world.spawn_actor(
            lane_invasion_bp, carla.Transform(), attach_to=self.vehicle
        )
        self.lane_invasion_sensor.listen(lambda event: self._on_lane_invasion(event))

        # Initialization
        self.episode_step = 0
        self.current_waypoint_index = 0
        self.route_completion = 0.0
        self.lane_invasion_history = []
        self.infractions = {k: 0 for k in self.infractions}
        self.prev_driving_score = 0.0
        self.current_image = None
        self.negative_reward_count = 0
        self.min_distance_to_route = 0.0
        self.vehicle_physics = VehiclePhysics.create()
        self.physics_history = []

        # Get the first frame
        while True:
            self.world.tick()
            if self.current_image is not None:
                break
            time.sleep(0.1)

        self._update_spectator()

        return self.current_image.copy(), {"task_prompt": self.prompt}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, any]]:
        # Apply action: action[0]=steer, action[1]=gas_or_brake (same as CarRacing)
        self.current_action = np.array(action, dtype=np.float32)
        steer = float(np.clip(action[0], -1.0, 1.0))
        gas_or_brake = float(np.clip(action[1], -1.0, 1.0))
        throttle = max(gas_or_brake, 0.0)
        brake = 0.0

        control = carla.VehicleControl(
            steer=steer, throttle=throttle, brake=brake, hand_brake=False, manual_gear_shift=False
        )
        self.vehicle.apply_control(control)

        self.world.tick()
        self._update_spectator()

        # Update route tracking
        self._update_route_progress()

        # Termination conditions
        has_collision = (
            self.infractions["collision_pedestrian"] > 0
            or self.infractions["collision_vehicle"] > 0
            or self.infractions["collision_static"] > 0
        )
        terminated = has_collision or self.route_completion >= 1.0

        # Calculate reward
        reward = self._compute_reward(has_collision)

        # Negative reward count
        if reward < 0:
            self.negative_reward_count += 1
        else:
            self.negative_reward_count = 0

        truncated = (
            self.episode_step >= self.max_episode_steps
            or self.negative_reward_count >= 100
            or self.min_distance_to_route >= 30.0
        )

        self.episode_step += 1

        assert self.current_image is not None

        # Concatenate observation image and route image
        camera_img = self.current_image.copy()  # (C, H, W)
        route_img = self._generate_route_image(show_text=False)  # (H, W, 3)

        # Resize route image to 1/4 size of camera image
        route_h = self.image_size[1] // 4
        route_w = self.image_size[0] // 4
        route_img_resized = cv2.resize(route_img, (route_w, route_h))

        # Convert camera image from (C, H, W) -> (H, W, C)
        camera_img_hwc = (camera_img.transpose(1, 2, 0) * 255).astype(np.uint8)

        # Place route image at bottom-right
        camera_img_hwc[-route_h:, -route_w:] = route_img_resized

        # Convert back (H, W, C) -> (C, H, W) and normalize
        obs = camera_img_hwc.transpose(2, 0, 1).astype(np.float32) / 255.0

        # Update various physics quantities
        self.vehicle_physics.update(self.vehicle, throttle, brake, steer)

        # Record history
        self.physics_history.append(copy.deepcopy(self.vehicle_physics))
        if len(self.physics_history) > self.history_length:
            self.physics_history = self.physics_history[-self.history_length :]

        info = {
            "task_prompt": self.prompt,
            "route_completion": self.route_completion,
            "infractions": self.infractions.copy(),
            "velocity": self.vehicle_physics.velocity,
            "velocity_kph": self.vehicle_physics.velocity_kph,
            "acceleration": self.vehicle_physics.acceleration,
            "lon_acceleration": self.vehicle_physics.lon_acceleration,
            "lat_acceleration": self.vehicle_physics.lat_acceleration,
            "jerk": self.vehicle_physics.jerk,
            "lon_jerk": self.vehicle_physics.lon_jerk,
            "angular_velocity": self.vehicle_physics.angular_velocity,
            "angular_acceleration": self.vehicle_physics.angular_acceleration,
        }

        return obs, reward, terminated, truncated, info

    def _generate_route_image(self, show_text: bool) -> np.ndarray:
        """Generate route image (vehicle always centered and facing up)

        Args:
            show_text: If True, display information as text in top-left
        """
        map_size = 512
        render_img = np.ones((map_size, map_size, 3), dtype=np.uint8) * 200

        # Vehicle current position and orientation
        vehicle_loc = self.vehicle.get_location()
        vehicle_transform = self.vehicle.get_transform()
        vehicle_yaw = np.radians(vehicle_transform.rotation.yaw)

        # Draw route (convert to vehicle coordinate system and rotate)
        for i in range(len(self.route_waypoints) - 1):
            wp1 = self.route_waypoints[i]
            wp2 = self.route_waypoints[i + 1]

            # Convert world coordinates to vehicle coordinate system
            x1, y1 = self._world_to_vehicle_coords(wp1, vehicle_loc, vehicle_yaw, map_size)
            x2, y2 = self._world_to_vehicle_coords(wp2, vehicle_loc, vehicle_yaw, map_size)

            # Draw route
            color = (255, 0, 0) if i < self.current_waypoint_index else (100, 100, 255)
            cv2.line(render_img, (x1, y1), (x2, y2), color, 20)

        # Draw vehicle as upward-facing triangle at center
        center_x, center_y = map_size // 2, map_size // 2
        triangle_size = 15

        # Upward-facing (yaw=0) triangle
        front_x = center_x
        front_y = center_y - triangle_size
        left_x = int(center_x - triangle_size * 0.5)
        left_y = int(center_y + triangle_size * 0.5)
        right_x = int(center_x + triangle_size * 0.5)
        right_y = int(center_y + triangle_size * 0.5)

        triangle_pts = np.array(
            [[front_x, front_y], [left_x, left_y], [right_x, right_y]], np.int32
        )
        cv2.fillPoly(render_img, [triangle_pts], (0, 255, 0))

        # Display information as text in top-left (only when show_text is True)
        if show_text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_color = (0, 0, 0)
            line_height = 20
            x_offset = 10
            y_offset = 20

            texts = [
                f"Route Completion: {self.route_completion * 100:.1f}%",
                f"Velocity: {self.vehicle_physics.velocity_kph:.1f} km/h",
                f"Acceleration: {self.vehicle_physics.acceleration:.2f} m/s^2",
                f"Jerk: {self.vehicle_physics.jerk:.2f} m/s^3",
                f"Angular Vel: {self.vehicle_physics.angular_velocity:.2f} rad/s",
                f"Angular Accel: {self.vehicle_physics.angular_acceleration:.2f} rad/s^2",
                f"Steer: {self.current_action[0]:+.3f}",
                f"Gas/Brake: {self.current_action[1]:+.3f}",
            ]

            for i, text in enumerate(texts):
                y_pos = y_offset + i * line_height
                cv2.putText(
                    render_img,
                    text,
                    (x_offset, y_pos),
                    font,
                    font_scale,
                    text_color,
                    font_thickness,
                )

        return render_img

    def render(self) -> np.ndarray | None:
        """Return third-person camera image with time-series graphs overlay."""
        if self.render_mode != "rgb_array":
            return None
        if self.third_person_image is None:
            return None

        img = self.third_person_image.copy()
        overlay_vehicle_graphs(img, self.physics_history)
        return img

    def close(self):
        if self.camera_sensor is not None:
            self.camera_sensor.destroy()
        if self.third_person_camera is not None:
            self.third_person_camera.destroy()
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

    def _generate_route(self) -> list:
        """Generate random route"""
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

            # Randomly select next waypoint (branch at intersections)
            current_wp = np.random.choice(next_wps)
            route_waypoints.append(current_wp)

            if len(route_waypoints) > 1:
                prev_loc = route_waypoints[-2].transform.location
                curr_loc = current_wp.transform.location
                distance += prev_loc.distance(curr_loc)

        start_pose = route_waypoints[0].transform

        # Interpolate to 1000 points (using scipy.interpolate)
        num_interp_points = 1000
        route_locations = np.array(
            [
                [wp.transform.location.x, wp.transform.location.y, wp.transform.location.z]
                for wp in route_waypoints
            ]
        )
        t_original = np.linspace(0, 1, len(route_locations))
        t_interp = np.linspace(0, 1, num_interp_points)

        interpolator = scipy.interpolate.interp1d(
            t_original, route_locations, axis=0, kind="linear"
        )
        interpolated_locations = interpolator(t_interp)
        interpolated_waypoints = []
        for loc in interpolated_locations:
            interpolated_waypoints.append(carla.Location(*loc))

        return interpolated_waypoints, start_pose

    def _update_route_progress(self):
        """Update route progress"""
        vehicle_loc = self.vehicle.get_location()

        # Find closest waypoint (within a certain range from current position)
        min_dist = float("inf")
        closest_idx = self.current_waypoint_index

        search_end = min(self.current_waypoint_index + 20, len(self.route_waypoints))
        for i in range(self.current_waypoint_index, search_end):
            dist = vehicle_loc.distance(self.route_waypoints[i])
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        self.current_waypoint_index = closest_idx
        self.route_completion = min(1.0, closest_idx / max(1, len(self.route_waypoints) - 1))
        self.min_distance_to_route = min_dist

    def _compute_reward(self, has_collision: bool) -> float:
        """
        Reward calculation combining route progress, collision penalty, and comfort.

        - Collision: -1.0 (terminal)
        - Route progress: delta of driving score
        - Comfort: penalty for exceeding acceleration/jerk/yaw thresholds
        """
        if has_collision:
            return -1.0

        # Route Completion (0.0 ~ 100.0)
        route_completion_score = self.route_completion * 100.0

        # Driving Score (no infraction penalty needed here since collision returns early)
        driving_score = route_completion_score

        # Reward is the difference from previous
        reward = driving_score - self.prev_driving_score
        self.prev_driving_score = driving_score

        # Give -0.1 if reward is 0.0 (no progress)
        if reward == 0.0:
            reward = -0.1

        # Comfort penalty: count how many comfort metrics are out of bounds
        comfort_penalty = self._compute_comfort_penalty()
        reward += comfort_penalty

        return reward

    def _compute_comfort_penalty(self) -> float:
        """
        Compute comfort penalty based on vehicle dynamics thresholds.
        Penalty scales with how much the value exceeds the bound:
          penalty = (value - bound) / |bound| + 1  (0 if within bounds)
        """
        physics = self.vehicle_physics
        metrics = [
            (physics.lat_acceleration, -COMFORT_MAX_ABS_LAT_ACCEL, COMFORT_MAX_ABS_LAT_ACCEL),
            (physics.lon_acceleration, COMFORT_MIN_LON_ACCEL, COMFORT_MAX_LON_ACCEL),
            (physics.jerk, -COMFORT_MAX_ABS_MAG_JERK, COMFORT_MAX_ABS_MAG_JERK),
            (physics.lon_jerk, -COMFORT_MAX_ABS_LON_JERK, COMFORT_MAX_ABS_LON_JERK),
            (physics.angular_velocity, -COMFORT_MAX_ABS_YAW_RATE, COMFORT_MAX_ABS_YAW_RATE),
            (physics.angular_acceleration, -COMFORT_MAX_ABS_YAW_ACCEL, COMFORT_MAX_ABS_YAW_ACCEL),
        ]
        penalty = 0.0
        for value, lo, hi in metrics:
            if value > hi:
                penalty += (value - hi) / hi + 1
            elif value < lo:
                penalty += (lo - value) / abs(lo) + 1
        return -0.0 * penalty

    def _world_to_vehicle_coords(
        self,
        world_loc: carla.Location,
        vehicle_loc: carla.Location,
        vehicle_yaw: float,
        map_size: int,
    ) -> tuple[int, int]:
        """Convert world coordinates to vehicle coordinate system (vehicle centered facing up)"""
        # Relative position from vehicle
        dx = world_loc.x - vehicle_loc.x
        dy = world_loc.y - vehicle_loc.y

        # Rotate according to vehicle orientation (so vehicle faces up)
        # Add 90 degree counter-clockwise rotation
        adjusted_yaw = -vehicle_yaw + np.pi / 2
        cos_yaw = np.cos(adjusted_yaw)
        sin_yaw = np.sin(adjusted_yaw)
        rotated_x = dx * cos_yaw - dy * sin_yaw
        rotated_y = dx * sin_yaw + dy * cos_yaw

        # Convert to pixel coordinates (flip left-right)
        scale = 0.5  # meters/pixel
        pixel_x = int(-rotated_x / scale + map_size // 2)
        pixel_y = int(-rotated_y / scale + map_size // 2)

        return pixel_x, pixel_y

    def _update_spectator(self):
        """Move spectator camera to follow the vehicle from behind and above."""
        vehicle_transform = self.vehicle.get_transform()
        forward = vehicle_transform.get_forward_vector()
        spectator_location = vehicle_transform.location + carla.Location(
            x=-8.0 * forward.x, y=-8.0 * forward.y, z=5.0
        )
        spectator_transform = carla.Transform(
            spectator_location,
            carla.Rotation(pitch=-20.0, yaw=vehicle_transform.rotation.yaw),
        )
        self.world.get_spectator().set_transform(spectator_transform)

    def _process_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, [2, 1, 0]]
        array = array.transpose(2, 0, 1).astype(np.float32) / 255.0
        self.current_image = array

    def _process_third_person_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, [2, 1, 0]]
        self.third_person_image = array

    def _on_collision(self, event):
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
