# environment.py
# CARLA Gym environment — SmartDrive + BLIP-FusePPO hybrid.
#
# KEY FIXES in this version:
#
#   FIX-1 (reward signature):
#       compute_reward() now receives distance_px and min_lidar_m.
#       Previously the LiDAR and pixel-lane terms in parameters.py were
#       never passed in, so only the terminal penalty fired correctly.
#
#   FIX-6 (augmented reward):
#       The augmented transition now recomputes reward with the flipped
#       observation instead of cloning the real reward. Previously every
#       real transition was counted twice in the value function, inflating
#       V(s) estimates and destabilising training.
#       BLIP is already regenerated for the flipped image in _build_aug_obs()
#       — that part was correct and is preserved.
#
#   FIX-BLIP-warmup:
#       BLIP state embedding is always included in obs (state role).
#       The warmup gate is removed — it was suppressing the BLIP bonus in
#       the reward, which no longer exists (FIX-1 in reward.py).
#       BLIP embeddings are meaningful from episode 1 (real CARLA frames).
#
#   Previously documented fixes (FIX-A through FIX-E) are preserved.

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
from reward       import compute_reward, reward_terms
from parameters   import (
    IM_HEIGHT, IM_WIDTH,
    LIDAR_DIM, PID_DIM, NAV_DIM,
    BLIP_EMBEDDING_DIM,
    OBS_KEY_IMAGE, OBS_KEY_BLIP, OBS_KEY_LIDAR, OBS_KEY_PID, OBS_KEY_NAV,
    TOWN, CAR_NAME, NUMBER_OF_PEDESTRIAN, VISUAL_DISPLAY,
    MAX_SPEED, MIN_SPEED, TARGET_SPEED,
    BLIP_UPDATE_INTERVAL,
    LANE_RESET_DIST, LIDAR_DFAIL,
    LIDAR_BELOW_THRESH_COUNT, LIDAR_WINDOW_STEPS,
    MAX_DISTANCE_FROM_CENTER,
    REWARD_FAIL, REWARD_TERMINAL, EPISODE_LENGTH,
    LANE_NORM,
)


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


class CarlaEnv(gym.Env):
    """
    Gym-compatible CARLA environment for SB3 MultiInputPolicy.

    Observation space (Dict):
        image          -> (IM_HEIGHT, IM_WIDTH, 3)  float32  [0, 1]
        blip           -> (768,)                    float32
        lidar          -> (180,)                    float32  [0, 1]
        pid_correction -> (1,)                      float32
        navigation     -> (5,)                      float32

    Action space:
        Box(-1, 1, (2,))  ->  [steer, throttle_raw]
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, client, world, checkpoint_frequency=50):
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
            OBS_KEY_BLIP: spaces.Box(
                -1.0, 1.0, (BLIP_EMBEDDING_DIM,), dtype=np.float32),
            OBS_KEY_LIDAR: spaces.Box(
                0.0, 1.0, (LIDAR_DIM,), dtype=np.float32),
            OBS_KEY_PID: spaces.Box(
                -1.0, 1.0, (PID_DIM,), dtype=np.float32),
            OBS_KEY_NAV: spaces.Box(
                -5.0, 5.0, (NAV_DIM,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

        # ── Sensor handles ───────────────────────────────────────────
        self.camera_obj     = None
        self.env_camera_obj = None
        self.collision_obj  = None
        self.lidar_obj      = None
        self.sensor_list    = []
        self.actor_list     = []
        self.walker_list    = []

        # ── Helpers ──────────────────────────────────────────────────
        self.blip_encoder = BLIPEncoder()
        self.pid          = PIDController()

        # ── State ────────────────────────────────────────────────────
        self._episode_count  = 0
        self._frame_counter  = 0
        self._last_blip_emb  = np.zeros(BLIP_EMBEDDING_DIM, dtype=np.float32)
        self._last_aug_blip  = np.zeros(BLIP_EMBEDDING_DIM, dtype=np.float32)
        self._last_image_rgb = None
        self._last_pid_val   = 0.0
        self._last_distance_px = 0.0
        self._last_min_lidar_m = 100.0

        self.vehicle                   = None
        self.route_waypoints           = []
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

        self._lidar_below_count  = 0
        self._lidar_window_steps = 0

        self._aug_phase           = 0
        self._last_aug_transition = None

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
            self._last_pid_val       = 0.0
            self._last_distance_px   = 0.0
            self._last_min_lidar_m   = 100.0

            self._last_blip_emb  = np.zeros(BLIP_EMBEDDING_DIM, dtype=np.float32)
            self._last_aug_blip  = np.zeros(BLIP_EMBEDDING_DIM, dtype=np.float32)
            self._last_image_rgb = None

            # ── Spawn vehicle ─────────────────────────────────────────
            vehicle_bp = self._get_vehicle_bp(CAR_NAME)
            if self.town == 'Town02':
                self.total_distance = 500
                spawn_idx = 30
            elif self.town == 'Town07':
                self.total_distance = 750
                spawn_idx = 20
            else:
                self.total_distance = 500
                spawn_idx = 12

            # Get the fixed spawn point transform
            spawn_transform = self.map.get_spawn_points()[spawn_idx]

            # Project it onto the nearest driving waypoint so the car
            # starts exactly on the lane centre facing the road direction.
            # This eliminates wall collisions from awkward spawn angles.
            spawn_wp = self.map.get_waypoint(
                spawn_transform.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            spawn_tf = spawn_wp.transform
            spawn_tf.location.z = spawn_transform.location.z + 0.3

            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_tf)
            if self.vehicle is None:
                # Fallback: use raw spawn point
                spawn_transform.location.z += 0.3
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
                spawn_wp = self.map.get_waypoint(
                    spawn_transform.location,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
            if self.vehicle is None:
                raise RuntimeError("Failed to spawn vehicle")
            self.actor_list.append(self.vehicle)

            # ── Attach sensors ────────────────────────────────────────
            self.camera_obj = CameraSensor(self.vehicle)
            while len(self.camera_obj.front_camera) == 0:
                time.sleep(0.001)
            self.sensor_list.append(self.camera_obj.sensor)

            if self.display_on:
                self.env_camera_obj = CameraSensorEnv(self.vehicle)
                self.sensor_list.append(self.env_camera_obj.sensor)

            self.collision_obj = CollisionSensor(self.vehicle)
            self.collision_history = self.collision_obj.collision_data
            self.sensor_list.append(self.collision_obj.sensor)

            self.lidar_obj = LiDARSensor(self.vehicle)
            self.sensor_list.append(self.lidar_obj.sensor)

            # ── Episode state ─────────────────────────────────────────
            self.timesteps             = 0
            self.throttle              = 0.0
            self.previous_steer        = 0.0
            self.velocity              = 0.0
            self.distance_from_center  = 0.0
            self.angle                 = 0.0
            self.center_lane_deviation = 0.0
            self.distance_covered      = 0.0
            # episode_start_time set after throttle impulse below

            # ── Route waypoints ───────────────────────────────────────
            if self.fresh_start:
                self.current_waypoint_index = 0
                self.route_waypoints = []
                # FIX-WAYPOINT: start route from the actual spawn waypoint,
                # not from get_waypoint(vehicle.get_location()) which can
                # project to the wrong lane and cause immediate lane_exit.
                wp = spawn_wp
                self.route_waypoints.append(wp)
                for x in range(self.total_distance):
                    nexts = wp.next(1.0)
                    if not nexts:
                        break
                    if self.town == 'Town02':
                        nxt = nexts[-1] if x > 100 else nexts[0]
                    elif self.town == 'Town07':
                        nxt = nexts[-1] if x < 650 else nexts[0]
                    else:
                        nxt = nexts[-1] if x < 300 else nexts[0]
                    self.route_waypoints.append(nxt)
                    wp = nxt
            else:
                wp = self.route_waypoints[
                    self.checkpoint_waypoint_index % len(self.route_waypoints)]
                self.vehicle.set_transform(wp.transform)
                self.current_waypoint_index = self.checkpoint_waypoint_index

            # Wait for physics to settle before starting episode.
            time.sleep(1.0)

            # FIX-SPEED: Apply initial throttle impulse so the car is
            # already moving at the start of step 1. Without this, the
            # car sits at v=0 and r_speed stays very negative for the
            # first ~20 steps while the agent slowly learns to throttle.
            # 0.7 throttle for 0.8s gets the car to ~10-15 km/h at step 1.
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=0.7, steer=0.0, brake=0.0))
            time.sleep(0.8)

            # Clear collision history AFTER settling — any spawn bumps
            # from the physics engine initialising are discarded.
            self.collision_history.clear()
            # Reset episode start time AFTER impulse so 30s grace period
            # is measured from when the agent actually takes control.
            self.episode_start_time = time.time()

            obs = self._build_obs()
            return obs

        except Exception as e:
            print(f"[Env.reset] error: {e}")
            self._destroy_actors()
            return self._blank_obs()

    # ------------------------------------------------------------------ #
    #  Step                                                               #
    # ------------------------------------------------------------------ #

    def step(self, action):
        try:
            # ── Augmented phase — return pre-built transition ─────────
            if self._aug_phase == 1:
                self._aug_phase = 0
                return self._last_aug_transition

            # ── Apply action ─────────────────────────────────────────
            self.timesteps      += 1
            self.fresh_start     = False
            self._frame_counter += 1

            steer    = float(np.clip(action[0], -1.0, 1.0))
            throttle = float(np.clip((action[1] + 1.0) / 2.0, 0.0, 1.0))

            # FIX-SMOOTH: Removed 0.9/0.1 action smoothing.
            # With smoothing, throttle after 1 step = 0.1 * action only.
            # After 10 steps with action=1.0, throttle only reaches 0.65.
            # This means the car barely accelerates, r_speed stays very
            # negative, and the agent never learns that throttle = speed.
            # SmartDrive used smoothing but had a 1200-episode budget.
            # With 400 episodes we need direct, responsive control.
            self.vehicle.apply_control(carla.VehicleControl(
                steer    = steer,
                throttle = throttle,
            ))
            self.previous_steer = steer
            self.throttle       = throttle

            if self.vehicle.is_at_traffic_light():
                tl = self.vehicle.get_traffic_light()
                if tl.get_state() == carla.TrafficLightState.Red:
                    tl.set_state(carla.TrafficLightState.Green)

            # ── Vehicle state ─────────────────────────────────────────
            vel           = self.vehicle.get_velocity()
            self.velocity = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) * 3.6
            self.collision_history = self.collision_obj.collision_data
            location = self.vehicle.get_location()

            # ── Waypoint tracking ─────────────────────────────────────
            if not self.route_waypoints:
                raise RuntimeError("route_waypoints not initialised")
            wi = self.current_waypoint_index
            # FIX-TRACK: cap loop at 50, not len(route_waypoints).
            # Previous loop over full 500-waypoint list had an exit
            # condition that meant only 1-2 waypoints advanced per step.
            # At 20 km/h with 1m spacing, car can pass up to ~5-10
            # waypoints per step. 50 gives a safe ceiling.
            for _ in range(50):
                nwi = wi + 1
                if nwi >= len(self.route_waypoints):
                    break
                wp     = self.route_waypoints[nwi]
                wp_fwd = self._vec(wp.transform.get_forward_vector())[:2]
                to_veh = self._vec(location - wp.transform.location)[:2]
                if np.dot(wp_fwd, to_veh) > 0.0:
                    wi += 1
                else:
                    break
            self.current_waypoint_index = wi
            cur_wp  = self.route_waypoints[wi % len(self.route_waypoints)]
            next_wp = self.route_waypoints[min(wi + 1, len(self.route_waypoints) - 1)]

            self.distance_from_center = self._dist_to_line(
                self._vec(cur_wp.transform.location),
                self._vec(next_wp.transform.location),
                self._vec(location),
            )
            self.center_lane_deviation += self.distance_from_center

            fwd    = self._vec(self.vehicle.get_velocity())
            wp_fwd = self._vec(cur_wp.transform.rotation.get_forward_vector())
            self.angle = self._signed_angle_diff(fwd, wp_fwd)

            # ── LiDAR readings ────────────────────────────────────────
            lidar_norm     = self.lidar_obj.range_data        # (180,) in [0,1]
            # Sensor normalises to [0,1] over 100 m max range
            min_lidar_m    = float(np.min(lidar_norm)) * 100.0
            self._last_min_lidar_m = min_lidar_m

            # ── Lane pixel distance (Hough, for reward terms) ─────────
            distance_px = self._compute_lane_distance_px()
            self._last_distance_px = distance_px

            # ── LiDAR termination window ──────────────────────────────
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

            # FIX-GRACE: Skip all termination checks for first 10 steps.
            # The spawn + physics settle can produce:
            #   - transient distance_from_center > 3m before car aligns
            #   - Hough returning bad distance_px on first frames
            #   - collision impulse from ground contact
            # 10 steps at 20km/h = ~1m — negligible route progress lost.
            early_grace = self.timesteps <= 10

            if not early_grace and len(self.collision_history) != 0:
                done = failed = True
            elif not early_grace and self.distance_from_center > MAX_DISTANCE_FROM_CENTER:
                done = failed = True
            elif not early_grace and abs(distance_px) > LANE_RESET_DIST:
                done = failed = True
            elif lidar_done:
                done = failed = True
            elif (self.episode_start_time + 30 < time.time()
                  and self.velocity < MIN_SPEED):
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

            # ── Checkpoint update (SmartDrive strategy) ───────────────
            if self.checkpoint_frequency is not None:
                self.checkpoint_waypoint_index = (
                    self.current_waypoint_index // self.checkpoint_frequency
                ) * self.checkpoint_frequency

            # ── Reward (FIX-1: pass distance_px + min_lidar_m) ────────
            # Compute individual terms first (for CSV diagnostic logging)
            terms = reward_terms(
                distance_px  = distance_px,
                min_lidar_m  = min_lidar_m,
                velocity_kmh = self.velocity,
            )
            reward = compute_reward(
                distance_from_center_m = self.distance_from_center,
                velocity_kmh           = self.velocity,
                angle_rad              = self.angle,
                done                   = done,
                failed                 = failed,
                distance_px            = distance_px,
                min_lidar_m            = min_lidar_m,
            )

            # Determine done_reason for CSV logging
            if failed:
                if len(self.collision_history) != 0:
                    done_reason = "collision"
                elif self.distance_from_center > MAX_DISTANCE_FROM_CENTER:
                    done_reason = "lane_exit_metric"
                elif abs(distance_px) > LANE_RESET_DIST:
                    done_reason = "lane_exit_px"
                elif lidar_done:
                    done_reason = "lidar"
                else:
                    done_reason = "other_fail"
            elif done:
                if self.current_waypoint_index >= len(self.route_waypoints) - 2:
                    done_reason = "destination"
                else:
                    done_reason = "timeout"
            else:
                done_reason = ""

            # ── Build real observation ────────────────────────────────
            obs = self._build_obs()

            # ── Build augmented transition (FIX-6: recompute reward) ──
            aug_obs    = self._build_aug_obs()
            aug_action = np.array([-action[0], action[1]], dtype=np.float32)

            # FIX-6: augmented reward recomputed for FLIPPED observation.
            # Previously the real reward was cloned — doubling every reward
            # signal in the value function. Now we recompute properly.
            # The flipped image has the same distance_from_center and velocity,
            # but the steering sign is inverted — reflected in aug_action.
            # distance_px sign is also flipped for the mirrored lane position.
            aug_reward = compute_reward(
                distance_from_center_m = self.distance_from_center,
                velocity_kmh           = self.velocity,
                angle_rad              = -self.angle,           # mirrored
                done                   = done,
                failed                 = failed,
                distance_px            = -distance_px,          # mirrored
                min_lidar_m            = min_lidar_m,           # same sensor
            )

            self._last_aug_transition = (
                aug_obs, aug_reward, done,
                {"augmented_action": aug_action},
            )
            self._aug_phase = 1

            if done:
                self._episode_count += 1
                self.center_lane_deviation /= max(self.timesteps, 1)
                self.distance_covered = abs(
                    self.current_waypoint_index
                    - self.checkpoint_waypoint_index)
                self._destroy_actors()
                self._aug_phase = 0

            info = {
                "distance_covered":      self.distance_covered,
                "center_lane_deviation": self.center_lane_deviation,
                "episode_count":         self._episode_count,
                "done_reason":           done_reason,
                # Per-term reward breakdown for CSV logger
                "r_lane":                terms["r_lane"],
                "r_lidar":               terms["r_lidar"],
                "r_speed":               terms["r_speed"],
                "r_center":              terms["r_center"],
            }
            return obs, reward, done, info

        except Exception as e:
            print(f"[Env.step] error: {e}")
            self._destroy_actors()
            return self._blank_obs(), REWARD_FAIL, True, {}

    # ------------------------------------------------------------------ #
    #  Observation builders                                               #
    # ------------------------------------------------------------------ #

    def _build_obs(self):
        while len(self.camera_obj.front_camera) == 0:
            time.sleep(0.0001)
        image_rgb = self.camera_obj.front_camera.pop(-1)
        self._last_image_rgb = image_rgb
        image_norm = image_rgb.astype(np.float32) / 255.0

        # BLIP — always included in state (no warmup gate — FIX-BLIP-warmup)
        if self._frame_counter % BLIP_UPDATE_INTERVAL == 0:
            self._last_blip_emb = self.blip_encoder.get_embedding(image_rgb)
        blip_emb = self._last_blip_emb.copy()

        lidar_vec = self.lidar_obj.range_data.copy()

        raw_pid = self.pid.compute(self._last_distance_px / 100.0)
        self._last_pid_val = float(np.clip(raw_pid, -1.0, 1.0))

        normalized_velocity             = self.velocity / max(TARGET_SPEED, 1e-6)
        normalized_distance_from_center = self.distance_from_center / 3.0
        normalized_angle                = abs(self.angle) / np.deg2rad(20)
        nav_vec = np.array([
            self.throttle,
            normalized_velocity,
            normalized_distance_from_center,
            normalized_angle,
            float(self.current_waypoint_index)
            / max(float(len(self.route_waypoints)), 1.0),
        ], dtype=np.float32)

        return {
            OBS_KEY_IMAGE: image_norm,
            OBS_KEY_BLIP:  blip_emb,
            OBS_KEY_LIDAR: lidar_vec,
            OBS_KEY_PID:   np.array([self._last_pid_val], dtype=np.float32),
            OBS_KEY_NAV:   nav_vec,
        }

    def _build_aug_obs(self):
        """
        Horizontally mirrored observation.
        Image flipped, LiDAR reversed, PID sign inverted.
        BLIP regenerated for the flipped image (paper Fig.3 requirement).
        """
        if self._last_image_rgb is None or self.lidar_obj is None:
            return self._blank_obs()

        aug_image_rgb  = cv2.flip(self._last_image_rgb, 1)
        aug_image_norm = aug_image_rgb.astype(np.float32) / 255.0

        # Regenerate BLIP for the flipped image per paper Fig.3
        if self._frame_counter % BLIP_UPDATE_INTERVAL == 0:
            self._last_aug_blip = self.blip_encoder.get_embedding(aug_image_rgb)
        aug_blip_emb = self._last_aug_blip.copy()

        aug_lidar = np.flip(self.lidar_obj.range_data.copy()).astype(np.float32)
        aug_pid   = float(np.clip(-self._last_pid_val, -1.0, 1.0))

        normalized_velocity             = self.velocity / max(TARGET_SPEED, 1e-6)
        normalized_distance_from_center = self.distance_from_center / 3.0
        normalized_angle                = abs(self.angle) / np.deg2rad(20)
        nav_vec = np.array([
            self.throttle,
            normalized_velocity,
            normalized_distance_from_center,
            normalized_angle,
            float(self.current_waypoint_index)
            / max(float(len(self.route_waypoints)), 1.0),
        ], dtype=np.float32)

        return {
            OBS_KEY_IMAGE: aug_image_norm,
            OBS_KEY_BLIP:  aug_blip_emb,
            OBS_KEY_LIDAR: aug_lidar,
            OBS_KEY_PID:   np.array([aug_pid], dtype=np.float32),
            OBS_KEY_NAV:   nav_vec,
        }

    def _blank_obs(self):
        return {
            OBS_KEY_IMAGE: np.zeros((IM_HEIGHT, IM_WIDTH, 3), dtype=np.float32),
            OBS_KEY_BLIP:  np.zeros(BLIP_EMBEDDING_DIM, dtype=np.float32),
            OBS_KEY_LIDAR: np.ones(LIDAR_DIM, dtype=np.float32),
            OBS_KEY_PID:   np.zeros(PID_DIM, dtype=np.float32),
            OBS_KEY_NAV:   np.zeros(NAV_DIM, dtype=np.float32),
        }

    # ------------------------------------------------------------------ #
    #  Lane detection (pixel distance, for reward only)                   #
    # ------------------------------------------------------------------ #

    def _compute_lane_distance_px(self):
        """
        Hough-based lane centre offset in pixels.
        Returns 0.0 if no lanes detected.
        """
        if self._last_image_rgb is None:
            return 0.0
        try:
            gray = cv2.cvtColor(self._last_image_rgb, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            h, w  = edges.shape
            roi   = edges[h // 2:, :]
            lines = cv2.HoughLinesP(roi, 1, np.pi / 180,
                                    threshold=30,
                                    minLineLength=20,
                                    maxLineGap=10)
            if lines is None:
                return 0.0
            xs = [(x1 + x2) / 2.0 for line in lines for x1, _, x2, _ in [line[0]]]
            if not xs:
                return 0.0
            lane_center = np.mean(xs)
            image_center = w / 2.0
            return float(lane_center - image_center)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------ #
    #  Geometry helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _vec(v):
        if hasattr(v, 'x'):
            return np.array([v.x, v.y, v.z], dtype=np.float64)
        return np.array(v, dtype=np.float64)

    @staticmethod
    def _dist_to_line(a, b, p):
        ab  = b - a
        ab_len = np.linalg.norm(ab[:2])
        if ab_len < 1e-6:
            return float(np.linalg.norm((p - a)[:2]))
        cross = abs(ab[0] * (a[1] - p[1]) - (a[0] - p[0]) * ab[1])
        return float(cross / ab_len)

    @staticmethod
    def _signed_angle_diff(fwd, wp_fwd):
        fwd_norm    = np.linalg.norm(fwd[:2])
        wp_fwd_norm = np.linalg.norm(wp_fwd[:2])
        if fwd_norm < 1e-6 or wp_fwd_norm < 1e-6:
            return 0.0
        cos_a = np.clip(np.dot(fwd[:2], wp_fwd[:2]) / (fwd_norm * wp_fwd_norm),
                        -1.0, 1.0)
        cross = fwd[0] * wp_fwd[1] - fwd[1] * wp_fwd[0]
        return float(math.copysign(math.acos(cos_a), cross))

    # ------------------------------------------------------------------ #
    #  Actor management                                                   #
    # ------------------------------------------------------------------ #

    def _get_vehicle_bp(self, name):
        bp = list(self.bp_lib.filter(name))
        if not bp:
            bp = list(self.bp_lib.filter('vehicle.*'))
        # Always take index 0 — random.choice was picking a different
        # car variant each episode, changing vehicle dynamics mid-training.
        return bp[0]

    def _destroy_actors(self):
        for sensor in self.sensor_list:
            try:
                if sensor and sensor.is_alive:
                    sensor.stop()
                    sensor.destroy()
            except Exception:
                pass
        self.sensor_list.clear()

        for actor in self.actor_list:
            try:
                if actor and actor.is_alive:
                    actor.destroy()
            except Exception:
                pass
        self.actor_list.clear()

        self.camera_obj     = None
        self.env_camera_obj = None
        self.collision_obj  = None
        self.lidar_obj      = None

    def _create_pedestrians(self):
        try:
            for _ in range(NUMBER_OF_PEDESTRIAN):
                sp = carla.Transform()
                loc = self.world.get_random_location_from_navigation()
                if loc is None:
                    continue
                sp.location = loc
                walker_bp = random.choice(
                    self.bp_lib.filter('walker.pedestrian.*'))
                ctrl_bp = self.bp_lib.find('controller.ai.walker')
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                if walker_bp.has_attribute('speed'):
                    walker_bp.set_attribute(
                        'speed',
                        walker_bp.get_attribute('speed').recommended_values[1])
                walker = self.world.try_spawn_actor(walker_bp, sp)
                if walker is None:
                    continue
                ctrl = self.world.spawn_actor(ctrl_bp, carla.Transform(), walker)
                self.walker_list.extend([ctrl.id, walker.id])
            actors = self.world.get_actors(self.walker_list)
            for i in range(0, len(actors), 2):
                actors[i].start()
                actors[i].go_to_location(
                    self.world.get_random_location_from_navigation())
        except Exception:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.walker_list])
            self.walker_list.clear()

    def close(self):
        self._destroy_actors()
        self.client.apply_batch(
            [carla.command.DestroyActor(x) for x in self.walker_list])
        self.walker_list.clear()