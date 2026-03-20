# environment.py
# CARLA Gym environment combining:
#   SmartDrive  → CARLA setup, waypoints, pedestrians,
#                 checkpoint reset, navigation obs
#   BLIP-FusePPO → Dict observation space, BLIP state,
#                  LiDAR, PID, hybrid reward, symmetric augmentation
#
# Returns Dict obs compatible with SB3 MultiInputPolicy.
# Internal vehicle bookkeeping uses plain numpy (no TF dependency here).

import os
import sys
import glob
import time
import math
import random
import numpy as np
import cv2
import pygame

import gym
from gym import spaces

try:
    sys.path.append(glob.glob('./carla/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from sensors      import (CameraSensor, CameraSensorEnv,
                           CollisionSensor, LiDARSensor, PIDController)
from blip_encoder import BLIPEncoder
from reward       import compute_reward
from parameters   import (
    IM_HEIGHT, IM_WIDTH,
    LIDAR_DIM, PID_DIM, NAV_DIM,
    OBS_KEY_IMAGE, OBS_KEY_LIDAR, OBS_KEY_PID, OBS_KEY_NAV,
    TOWN, CAR_NAME, NUMBER_OF_PEDESTRIAN, VISUAL_DISPLAY,
    MAX_SPEED, MIN_SPEED, TARGET_SPEED,
    BLIP_UPDATE_INTERVAL,
    LANE_RESET_DIST, LIDAR_DFAIL,
    LIDAR_BELOW_THRESH_COUNT, LIDAR_WINDOW_STEPS,
    REWARD_FAIL, EPISODE_LENGTH,
)


# ═══════════════════════════════════════════════════════════════════════ #
#  CARLA Client Connection                                               #
# ═══════════════════════════════════════════════════════════════════════ #

class ClientConnection:

    def __init__(self):
        self.host    = 'localhost'
        self.port    = 2000
        self.timeout = 20.0

    def setup(self):
        try:
            client = carla.Client(self.host, self.port)
            client.set_timeout(self.timeout)
            world  = client.load_world(TOWN)
            world.set_weather(carla.WeatherParameters.CloudyNoon)
            print("CARLA connection established.")
            return client, world
        except Exception as e:
            raise ConnectionError(f"CARLA connection failed: {e}")


# ═══════════════════════════════════════════════════════════════════════ #
#  CARLA Gym Environment                                                 #
# ═══════════════════════════════════════════════════════════════════════ #

class CarlaEnv(gym.Env):
    """
    Gym-compatible CARLA environment for SB3.

    Observation space  (Dict):
        image          → (IM_HEIGHT, IM_WIDTH, 3)  float32  [0, 1]
        lidar          → (180,)                    float32  [0, 1]
        pid_correction → (1,)                      float32
        navigation     → (5,)                      float32

    Action space:
        Box(-1, 1, (2,))  →  [steer, throttle_raw]
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, client, world, checkpoint_frequency: int = 100):
        super().__init__()

        self.client    = client
        self.world     = world
        self.bp_lib    = world.get_blueprint_library()
        self.map       = world.get_map()
        self.town      = TOWN
        self.display_on = VISUAL_DISPLAY
        self.checkpoint_frequency = checkpoint_frequency

        # ── Gym spaces ───────────────────────────────────────────────
        self.observation_space = spaces.Dict({
            OBS_KEY_IMAGE: spaces.Box(
                0.0, 1.0, (IM_HEIGHT, IM_WIDTH, 3), dtype=np.float32),
            OBS_KEY_LIDAR: spaces.Box(
                0.0, 1.0, (LIDAR_DIM,), dtype=np.float32),
            OBS_KEY_PID: spaces.Box(
                -1.0, 1.0, (PID_DIM,), dtype=np.float32),
            OBS_KEY_NAV: spaces.Box(
                -np.inf, np.inf, (NAV_DIM,), dtype=np.float32),
        })
        self.action_space = spaces.Box(
            -1.0, 1.0, (2,), dtype=np.float32)

        # ── Sensor handles ───────────────────────────────────────────
        self.camera_obj     = None
        self.env_camera_obj = None
        self.collision_obj  = None
        self.lidar_obj      = None
        self.sensor_list    = []
        self.actor_list     = []
        self.walker_list    = []

        # ── State helpers ────────────────────────────────────────────
        self.blip_encoder = BLIPEncoder()
        self.pid          = PIDController()

        # BLIP caching
        self._frame_counter  = 0
        self._last_blip_emb  = np.zeros(768, dtype=np.float32)
        self._last_aug_blip  = np.zeros(768, dtype=np.float32)
        self._last_image_rgb = None

        # Episode tracking
        self.vehicle                   = None
        self.route_waypoints           = None
        self.current_waypoint_index    = 0
        self.checkpoint_waypoint_index = 0
        self.fresh_start               = True
        self.total_distance            = 500
        self.timesteps                 = 0
        self.throttle                  = 0.0
        self.previous_steer            = 0.0
        self.velocity                  = 0.0
        self.distance_from_center      = 0.0
        self.angle                     = 0.0
        self.center_lane_deviation     = 0.0
        self.distance_covered          = 0.0
        self.episode_start_time        = time.time()
        self._last_distance_px         = 0.0

        # LiDAR termination counters
        self._lidar_below_count  = 0
        self._lidar_window_steps = 0

        # Augmentation phase flag (BLIP-FusePPO alternating step logic)
        self._aug_phase             = 0
        self._last_aug_transition   = None

        self._create_pedestrians()

    # ------------------------------------------------------------------ #
    #  Reset                                                              #
    # ------------------------------------------------------------------ #

    def reset(self):
        try:
            self._destroy_actors()
            self.pid.reset()
            self._frame_counter      = 0
            self._lidar_below_count  = 0
            self._lidar_window_steps = 0
            self._aug_phase          = 0

            # ── Spawn vehicle ─────────────────────────────────────────
            vehicle_bp = self._get_vehicle_bp(CAR_NAME)
            if self.town == 'Town02':
                transform = self.map.get_spawn_points()[30]
                self.total_distance = 500
            elif self.town == 'Town07':
                transform = self.map.get_spawn_points()[20]
                self.total_distance = 750
            else:
                transform = self.map.get_spawn_points()[12]
                self.total_distance = 500

            self.vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
            self.actor_list.append(self.vehicle)

            # ── Attach sensors ────────────────────────────────────────
            self.camera_obj = CameraSensor(self.vehicle)
            while len(self.camera_obj.front_camera) == 0:
                time.sleep(0.0001)
            self.sensor_list.append(self.camera_obj.sensor)

            if self.display_on:
                self.env_camera_obj = CameraSensorEnv(self.vehicle)
                self.sensor_list.append(self.env_camera_obj.sensor)

            self.collision_obj = CollisionSensor(self.vehicle)
            self.collision_history = self.collision_obj.collision_data
            self.sensor_list.append(self.collision_obj.sensor)

            self.lidar_obj = LiDARSensor(self.vehicle)
            self.sensor_list.append(self.lidar_obj.sensor)

            # ── Vehicle state ─────────────────────────────────────────
            self.timesteps             = 0
            self.throttle              = 0.0
            self.previous_steer        = 0.0
            self.velocity              = 0.0
            self.distance_from_center  = 0.0
            self.angle                 = 0.0
            self.center_lane_deviation = 0.0
            self.distance_covered      = 0.0
            self.episode_start_time    = time.time()
            self._last_distance_px     = 0.0

            # ── Waypoints  (SmartDrive route logic) ───────────────────
            if self.fresh_start:
                self.current_waypoint_index = 0
                self.route_waypoints = []
                wp = self.map.get_waypoint(
                    self.vehicle.get_location(),
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
                self.route_waypoints.append(wp)
                for x in range(self.total_distance):
                    if self.town == 'Town02':
                        nxt = wp.next(1.0)[-1] if x > 100 else wp.next(1.0)[0]
                    elif self.town == 'Town07':
                        nxt = wp.next(1.0)[-1] if x < 650 else wp.next(1.0)[0]
                    else:
                        nxt = wp.next(1.0)[-1] if x < 300 else wp.next(1.0)[0]
                    self.route_waypoints.append(nxt)
                    wp = nxt
            else:
                wp = self.route_waypoints[
                    self.checkpoint_waypoint_index % len(self.route_waypoints)]
                self.vehicle.set_transform(wp.transform)
                self.current_waypoint_index = self.checkpoint_waypoint_index

            time.sleep(0.5)
            self.collision_history.clear()

            return self._build_obs()

        except Exception as e:
            print(f"[Env.reset] error: {e}")
            self._destroy_actors()
            return self._blank_obs()

    # ------------------------------------------------------------------ #
    #  Step                                                               #
    # ------------------------------------------------------------------ #

    def step(self, action):
        """
        Implements BLIP-FusePPO's alternating augmentation phase:
          phase=0 → execute action, return original obs
          phase=1 → return stored augmented transition

        Returns
        -------
        obs    : Dict
        reward : float
        done   : bool
        info   : dict
        """
        try:
            # ── Augmented phase  (BLIP-FusePPO) ──────────────────────
            if self._aug_phase == 1:
                self._aug_phase = 0
                return self._last_aug_transition

            # ── Apply action ─────────────────────────────────────────
            self.timesteps      += 1
            self.fresh_start     = False
            self._frame_counter += 1

            steer    = float(np.clip(action[0], -1.0, 1.0))
            throttle = float(np.clip((action[1] + 1.0) / 2.0, 0.0, 1.0))
            self.vehicle.apply_control(carla.VehicleControl(
                steer    = self.previous_steer * 0.9 + steer    * 0.1,
                throttle = self.throttle       * 0.9 + throttle * 0.1,
            ))
            self.previous_steer = steer
            self.throttle       = throttle

            # Green light override  (SmartDrive)
            if self.vehicle.is_at_traffic_light():
                tl = self.vehicle.get_traffic_light()
                if tl.get_state() == carla.TrafficLightState.Red:
                    tl.set_state(carla.TrafficLightState.Green)

            # ── Vehicle state ─────────────────────────────────────────
            vel           = self.vehicle.get_velocity()
            self.velocity = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) * 3.6
            self.collision_history = self.collision_obj.collision_data
            location = self.vehicle.get_location()

            # ── Waypoint tracking  (SmartDrive) ───────────────────────
            wi = self.current_waypoint_index
            for _ in range(len(self.route_waypoints)):
                nwi = wi + 1
                wp  = self.route_waypoints[nwi % len(self.route_waypoints)]
                dot = np.dot(
                    self._vec(wp.transform.get_forward_vector())[:2],
                    self._vec(location - wp.transform.location)[:2],
                )
                if dot > 0.0:
                    wi += 1
                else:
                    break
            self.current_waypoint_index = wi
            cur_wp  = self.route_waypoints[wi % len(self.route_waypoints)]
            next_wp = self.route_waypoints[(wi + 1) % len(self.route_waypoints)]

            self.distance_from_center = self._dist_to_line(
                self._vec(cur_wp.transform.location),
                self._vec(next_wp.transform.location),
                self._vec(location),
            )
            self.center_lane_deviation += self.distance_from_center

            fwd    = self._vec(self.vehicle.get_velocity())
            wp_fwd = self._vec(cur_wp.transform.rotation.get_forward_vector())
            self.angle = self._angle_diff(fwd, wp_fwd)

            # ── LiDAR min distance (metres) ───────────────────────────
            lidar_norm  = self.lidar_obj.range_data       # (180,) in [0,1]
            min_lidar_m = float(np.min(lidar_norm)) * 100.0

            # ── Lane distance (pixels, for reward + PID) ──────────────
            distance_px = self._compute_lane_distance_px()
            self._last_distance_px = distance_px

            # ── LiDAR termination window  (BLIP-FusePPO) ─────────────
            if min_lidar_m < LIDAR_DFAIL:
                self._lidar_below_count += 1
            self._lidar_window_steps += 1
            lidar_done = False
            if self._lidar_window_steps >= LIDAR_WINDOW_STEPS:
                if self._lidar_below_count >= LIDAR_BELOW_THRESH_COUNT:
                    lidar_done = True
                self._lidar_below_count  = 0
                self._lidar_window_steps = 0

            # ── Terminal conditions ───────────────────────────────────
            done   = False
            failed = False

            if len(self.collision_history) != 0:
                done = failed = True
            elif abs(distance_px) > LANE_RESET_DIST:
                done = failed = True
            elif lidar_done:
                done = failed = True
            elif (self.episode_start_time + 10 < time.time()
                  and self.velocity < 1.0):
                done = failed = True
            elif self.velocity > MAX_SPEED:
                done = failed = True
            elif self.current_waypoint_index >= len(self.route_waypoints) - 2:
                done = True
                self.fresh_start = True
                if self.checkpoint_frequency is not None:
                    if self.checkpoint_frequency < self.total_distance // 2:
                        self.checkpoint_frequency += 2
                    else:
                        self.checkpoint_frequency = None
                        self.checkpoint_waypoint_index = 0
            elif self.timesteps >= 2e6:
                done = True

            reward = compute_reward(distance_px, min_lidar_m,
                                    self.velocity, failed)

            # ── Build original observation ────────────────────────────
            obs = self._build_obs()

            # ── Build augmented transition  (BLIP-FusePPO) ────────────
            aug_obs    = self._build_aug_obs()
            aug_action = np.array([-action[0], action[1]], dtype=np.float32)
            self._last_aug_transition = (aug_obs, reward, done,
                                         {"augmented_action": aug_action})
            self._aug_phase = 1   # next call returns augmented

            if done:
                self.center_lane_deviation /= max(self.timesteps, 1)
                self.distance_covered = abs(
                    self.current_waypoint_index
                    - self.checkpoint_waypoint_index)
                self._destroy_actors()
                self._aug_phase = 0   # reset on episode end

            info = {
                "distance_covered":    self.distance_covered,
                "center_lane_deviation": self.center_lane_deviation,
            }
            return obs, reward, done, info

        except Exception as e:
            print(f"[Env.step] error: {e}")
            self._destroy_actors()
            return self._blank_obs(), REWARD_FAIL, True, {}

    # ------------------------------------------------------------------ #
    #  Observation builders                                               #
    # ------------------------------------------------------------------ #

    def _build_obs(self) -> dict:
        """
        Returns the normal (non-augmented) Dict observation.
        BLIP is refreshed every BLIP_UPDATE_INTERVAL frames.
        """
        while len(self.camera_obj.front_camera) == 0:
            time.sleep(0.0001)
        image_rgb = self.camera_obj.front_camera.pop(-1)     # (H,W,3) uint8
        self._last_image_rgb = image_rgb

        # Normalise image to [0,1]
        image_norm = image_rgb.astype(np.float32) / 255.0   # (H,W,3)

        # BLIP embedding — refresh every K frames
        if self._frame_counter % BLIP_UPDATE_INTERVAL == 0:
            self._last_blip_emb = self.blip_encoder.get_embedding(image_rgb)

        # LiDAR  (180,) already in [0,1]
        lidar_vec = self.lidar_obj.range_data.copy()

        # PID correction
        pid_val = self.pid.compute(self._last_distance_px / 100.0)
        pid_val = float(np.clip(pid_val, -1.0, 1.0))

        # Navigation obs  (SmartDrive 5-dim)
        norm_vel   = self.velocity / max(TARGET_SPEED, 1e-6)
        norm_dist  = self.distance_from_center / 3.0
        norm_angle = abs(self.angle / np.deg2rad(20))
        nav_vec    = np.array(
            [self.throttle, self.velocity, norm_vel, norm_dist, norm_angle],
            dtype=np.float32,
        )

        return {
            OBS_KEY_IMAGE: image_norm,
            OBS_KEY_LIDAR: lidar_vec,
            OBS_KEY_PID:   np.array([pid_val], dtype=np.float32),
            OBS_KEY_NAV:   nav_vec,
        }

    def _build_aug_obs(self) -> dict:
        """
        Returns the symmetric augmented Dict observation:
            image  → horizontally flipped
            lidar  → reversed
            pid    → negated
            nav    → unchanged
        """
        if self._last_image_rgb is None:
            return self._blank_obs()

        aug_image_rgb = cv2.flip(self._last_image_rgb, 1)
        aug_image_norm = aug_image_rgb.astype(np.float32) / 255.0

        # BLIP for augmented image
        if self._frame_counter % BLIP_UPDATE_INTERVAL == 0:
            self._last_aug_blip = self.blip_encoder.get_embedding(aug_image_rgb)

        aug_lidar = np.flip(self.lidar_obj.range_data.copy()).astype(np.float32)

        pid_val = self.pid.compute(self._last_distance_px / 100.0)
        aug_pid = float(np.clip(-pid_val, -1.0, 1.0))

        norm_vel   = self.velocity / max(TARGET_SPEED, 1e-6)
        norm_dist  = self.distance_from_center / 3.0
        norm_angle = abs(self.angle / np.deg2rad(20))
        nav_vec    = np.array(
            [self.throttle, self.velocity, norm_vel, norm_dist, norm_angle],
            dtype=np.float32,
        )

        return {
            OBS_KEY_IMAGE: aug_image_norm,
            OBS_KEY_LIDAR: aug_lidar,
            OBS_KEY_PID:   np.array([aug_pid], dtype=np.float32),
            OBS_KEY_NAV:   nav_vec,
        }

    def _blank_obs(self) -> dict:
        return {
            OBS_KEY_IMAGE: np.zeros((IM_HEIGHT, IM_WIDTH, 3), dtype=np.float32),
            OBS_KEY_LIDAR: np.zeros(LIDAR_DIM, dtype=np.float32),
            OBS_KEY_PID:   np.zeros(PID_DIM,   dtype=np.float32),
            OBS_KEY_NAV:   np.zeros(NAV_DIM,   dtype=np.float32),
        }

    # ------------------------------------------------------------------ #
    #  Lane distance from image  (Hough Transform)                       #
    # ------------------------------------------------------------------ #

    def _compute_lane_distance_px(self) -> float:
        if (self.camera_obj is None
                or len(self.camera_obj.front_camera) == 0):
            return 0.0
        img   = self.camera_obj.front_camera[-1]
        gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 60, 15,
                                minLineLength=5, maxLineGap=100)
        if lines is None:
            return 0.0
        h, w  = img.shape[:2]
        cx    = w // 2
        lefts, rights = [], []
        for l in lines:
            x1, y1, x2, y2 = l[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if slope < -0.2:
                lefts.append((x1 + x2) // 2)
            elif slope > 0.2:
                rights.append((x1 + x2) // 2)
        lx  = int(np.mean(lefts))  if lefts  else 0
        rx  = int(np.mean(rights)) if rights else w
        mid = (lx + rx) // 2
        return float(mid - cx)

    # ------------------------------------------------------------------ #
    #  Pedestrians  (SmartDrive unchanged)                               #
    # ------------------------------------------------------------------ #

    def _create_pedestrians(self):
        try:
            spawn_pts = []
            for _ in range(NUMBER_OF_PEDESTRIAN):
                sp  = carla.Transform()
                loc = self.world.get_random_location_from_navigation()
                if loc:
                    sp.location = loc
                    spawn_pts.append(sp)
            for sp in spawn_pts:
                w_bp = random.choice(
                    self.bp_lib.filter('walker.pedestrian.*'))
                ctrl_bp = self.bp_lib.find('controller.ai.walker')
                if w_bp.has_attribute('is_invincible'):
                    w_bp.set_attribute('is_invincible', 'false')
                if w_bp.has_attribute('speed'):
                    w_bp.set_attribute(
                        'speed',
                        w_bp.get_attribute('speed').recommended_values[1])
                walker = self.world.try_spawn_actor(w_bp, sp)
                if walker:
                    ctrl = self.world.spawn_actor(
                        ctrl_bp, carla.Transform(), walker)
                    self.walker_list += [ctrl.id, walker.id]
            actors = self.world.get_actors(self.walker_list)
            for i in range(0, len(self.walker_list), 2):
                actors[i].start()
                actors[i].go_to_location(
                    self.world.get_random_location_from_navigation())
        except Exception:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.walker_list])

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _get_vehicle_bp(self, car_name):
        bp = self.bp_lib.filter(car_name)[0]
        if bp.has_attribute('color'):
            bp.set_attribute('color', random.choice(
                bp.get_attribute('color').recommended_values))
        return bp

    @staticmethod
    def _vec(v):
        if hasattr(v, 'x'):
            return np.array([v.x, v.y, v.z])
        return np.array(v)

    @staticmethod
    def _dist_to_line(a, b, p):
        ab = b - a
        ap = p - a
        t  = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-6)
        return float(np.linalg.norm(p - (a + t * ab)))

    @staticmethod
    def _angle_diff(v1, v2):
        cos = np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return math.acos(np.clip(cos, -1.0, 1.0))

    def _destroy_actors(self):
        if self.sensor_list:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.sensor_list])
            self.sensor_list.clear()
        if self.actor_list:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.actor_list])
            self.actor_list.clear()
        self.camera_obj = self.env_camera_obj = None
        self.collision_obj = self.lidar_obj   = None

    def render(self, mode='human'):
        pass

    def close(self):
        self._destroy_actors()
