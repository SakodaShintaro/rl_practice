# SPDX-License-Identifier: MIT
import copy
import time
from dataclasses import dataclass

import carla
import gymnasium as gym
import numpy as np

from vla_streaming_rl.envs.bench2drive_scenario_runtime import Bench2DriveRuntime
from vla_streaming_rl.envs.carla_obs import (
    CARLAObsConfig,
    RouteTracker,
    action_to_vehicle_control,
    compose_obs,
    make_action_space,
    make_obs_space,
)
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

    def __init__(
        self,
        route_xml: str | None,
        route_id: str | None,
    ):
        """Create the env.

        Args:
            route_xml: Path to a Bench2Drive route XML, or ``None`` to
                sample a random 200 m route in Town01 each episode. If a
                path is given, the town is read from the XML (all routes
                must share one town).
            route_id: Specific route id to pin to (only meaningful with
                ``route_xml`` set). ``None`` means "pick a random route
                from the XML each episode".
        """
        super().__init__()
        self.prompt = "Drive a car along a route in CARLA. Follow the planned route, obey traffic rules, and avoid collisions."

        # If a Bench2Drive route XML is given, the env runs the eval-equivalent
        # RouteScenario (ego spawn, BackgroundActivity NPC traffic, scripted
        # scenarios, parked vehicles) via Bench2DriveRuntime. All routes in
        # the file must share a town — switching towns mid-training would
        # force a ~30 s world reload on every reset.
        self._route_xml = route_xml
        self._route_id = route_id
        self._tm_port = 8000
        self.runtime: Bench2DriveRuntime | None = None
        town_name = "Town01"

        # Observation / action contracts come from carla_obs (shared with the
        # Bench2Drive eval agent so training and eval stay bit-aligned).
        self.obs_cfg = CARLAObsConfig()
        self.max_episode_steps = 1000
        self.render_mode = "rgb_array"
        self.fps = 10  # Simulation FPS

        self.observation_space = make_obs_space(self.obs_cfg)
        self.action_space = make_action_space()

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(120.0)

        # Build the runtime first (parses the XML eagerly) so we can read the
        # required town off it before deciding which world to load.
        if route_xml is not None:
            self.runtime = Bench2DriveRuntime(
                client=self.client,
                traffic_manager_port=self._tm_port,
                route_xml=route_xml,
                route_id=route_id,
            )
            town_name = self.runtime.town

        # reset_settings=False keeps the sync settings we apply right after.
        self.world = self.client.load_world(town_name, reset_settings=False)

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        self.world.apply_settings(settings)

        # Traffic Manager also needs to be in sync mode; otherwise world.tick
        # can deadlock on maps with background traffic.
        self.traffic_manager = self.client.get_trafficmanager(self._tm_port)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_hybrid_physics_mode(True)

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

        # Route tracker (initialized in reset)
        self.route_tracker: RouteTracker | None = None

        # Leaderboard evaluation metrics
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

        # Destroy our sensors before clearing the scenario — once their parent
        # actor is gone, sensor.destroy() can throw on already-dead handles.
        if self.camera_sensor is not None:
            self.camera_sensor.destroy()
        if self.third_person_camera is not None:
            self.third_person_camera.destroy()
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
        if self.lane_invasion_sensor is not None:
            self.lane_invasion_sensor.destroy()

        if self.runtime is not None:
            # The runtime owns the ego (RouteScenario spawns it) plus all NPCs
            # and parked vehicles. Tear them down and let it spawn a fresh set.
            self.runtime.cleanup()
            self.vehicle, route_locations = self.runtime.reset(self.world)
        else:
            if self.vehicle is not None:
                self.vehicle.destroy()
            route_locations, start_pose = self._sample_random_route()
            blueprint_library = self.world.get_blueprint_library()
            # role_name='hero' is required on large maps (Town13 etc.); without
            # it, attaching a camera to the vehicle crashes UE4 with SIGSEGV.
            vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
            vehicle_bp.set_attribute("role_name", "hero")
            spawn_transform = carla.Transform(
                carla.Location(
                    x=start_pose.location.x,
                    y=start_pose.location.y,
                    z=start_pose.location.z + 0.5,
                ),
                start_pose.rotation,
            )
            while True:
                try:
                    self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_transform)
                    break
                except RuntimeError:
                    time.sleep(0.1)
                    continue

        self.route_tracker = RouteTracker.from_raw_waypoints(route_locations, self.obs_cfg)

        blueprint_library = self.world.get_blueprint_library()

        # Camera sensor
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.obs_cfg.image_size[0]))
        camera_bp.set_attribute("image_size_y", str(self.obs_cfg.image_size[1]))
        camera_bp.set_attribute("fov", str(self.obs_cfg.fov))
        camera_transform = carla.Transform(
            carla.Location(x=self.obs_cfg.camera_x, z=self.obs_cfg.camera_z)
        )
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
        self.lane_invasion_history = []
        self.infractions = {k: 0 for k in self.infractions}
        self.prev_driving_score = 0.0
        self.current_image = None
        self.negative_reward_count = 0
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
        steer, throttle, brake = action_to_vehicle_control(self.current_action)

        control = carla.VehicleControl(
            steer=steer, throttle=throttle, brake=brake, hand_brake=False, manual_gear_shift=False
        )
        self.vehicle.apply_control(control)

        self.world.tick()

        # Tick the Bench2Drive scenario tree after the world advances so that
        # NPC commands issued this tick land on the next world.tick (matches
        # ScenarioManager._tick_scenario ordering).
        if self.runtime is not None:
            self.runtime.tick(self.world, self.episode_step)

        self._update_spectator()

        # Update route tracking
        self.route_tracker.update(self.vehicle.get_location())

        # Termination conditions
        has_collision = (
            self.infractions["collision_pedestrian"] > 0
            or self.infractions["collision_vehicle"] > 0
            or self.infractions["collision_static"] > 0
        )
        terminated = has_collision or self.route_tracker.route_completion >= 1.0

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
            or self.route_tracker.min_distance_to_route >= 30.0
        )

        self.episode_step += 1

        assert self.current_image is not None

        # Compose observation: camera RGB + route overlay in the bottom-right quarter.
        camera_hwc_uint8 = (self.current_image.transpose(1, 2, 0) * 255).astype(np.uint8)
        vehicle_yaw_rad = np.radians(self.vehicle.get_transform().rotation.yaw)
        overlay = self.route_tracker.render_overlay(self.vehicle.get_location(), vehicle_yaw_rad)
        obs = compose_obs(camera_hwc_uint8, overlay, self.obs_cfg)

        # Update various physics quantities
        self.vehicle_physics.update(self.vehicle, throttle, brake, steer)

        # Record history
        self.physics_history.append(copy.deepcopy(self.vehicle_physics))
        if len(self.physics_history) > self.history_length:
            self.physics_history = self.physics_history[-self.history_length :]

        info = {
            "task_prompt": self.prompt,
            "route_completion": self.route_tracker.route_completion,
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

        if self.runtime is not None:
            # Tears down the ego, NPCs, scripted scenario actors, parked vehicles.
            self.runtime.cleanup()
        elif self.vehicle is not None:
            self.vehicle.destroy()

        if self.world is not None:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            self.traffic_manager.set_synchronous_mode(False)

    def _sample_random_route(self) -> tuple[list[carla.Location], carla.Transform]:
        """Sample a random ~200 m route in the current map.

        Used only when no Bench2Drive XML is configured; the XML path goes
        through ``Bench2DriveRuntime`` which spawns the ego itself.
        """
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
        raw_locations = [wp.transform.location for wp in route_waypoints]
        return raw_locations, start_pose

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
        route_completion_score = self.route_tracker.route_completion * 100.0

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
