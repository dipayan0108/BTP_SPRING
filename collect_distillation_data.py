# collect_distillation_data.py
# =============================================================================
#
#  Distillation Dataset Collector — SmartDrive BTP
#
#  ROLE IN THE BTP PIPELINE  (PDF Fig 1 — Steps 1, 2, 3):
#  ─────────────────────────────────────────────────────────────────────────
#  Step 1  Data Collection  : CARLA autopilot drives smoothly
#  Step 2  Teacher Inference: Qwen-VL labels every frame
#  Step 3  Dataset Creation : save {image, state, teacher_label} to disk
#
#  ARCHITECTURE — Option C (Autopilot Drives, Teacher Labels):
#  ─────────────────────────────────────────────────────────────────────────
#  CARLA's built-in autopilot drives the ego vehicle.
#  Qwen-VL Teacher observes every frame and outputs what the CORRECT
#  action should be — this becomes the imitation learning label.
#  The teacher label is saved to CSV but NOT applied to the vehicle.
#
#  WHY THIS IS CORRECT:
#  ─────────────────────────────────────────────────────────────────────────
#  The teacher's job is to LABEL, not to drive.
#  Autopilot generates high-quality smooth trajectories.
#  Qwen provides semantic scene understanding and expert labels.
#  This decoupling maximises both trajectory quality and label quality.
#
#  WHAT GETS SAVED:
#  ─────────────────────────────────────────────────────────────────────────
#  Results_05/distillation_data/
#  ├── Episode_01/
#  │   ├── frame_0000.png   ← raw 160×80 semantic-seg image
#  │   ├── frame_0001.png
#  │   └── ...
#  ├── Episode_01_data.csv  ← one row per frame
#  ├── Episode_02/
#  ├── Episode_02_data.csv
#  └── collection_summary.csv
#
#  CSV COLUMNS:
#  ─────────────────────────────────────────────────────────────────────────
#  IDENTIFIERS:
#    frame_id, image_path
#
#  INPUT COLUMNS (student receives these at inference time):
#    velocity_kmh, dist_from_center, angle_rad,
#    steering_angle, throttle, nav_command
#
#  LABEL COLUMNS (imitation learning trains student to output these):
#    teacher_action_label, teacher_steer, teacher_throttle_raw, teacher_reason
#
#  AUTOPILOT COLUMNS (what actually happened — for analysis):
#    autopilot_steer, autopilot_throttle
#
#  FEEDBACK COLUMNS (analysis only):
#    reward, done, step_time_s
#
#  HOW TO RUN:
#  ─────────────────────────────────────────────────────────────────────────
#  Terminal 1:  CarlaUE4.exe -quality-level=Low -fps=20 -windowed
#  Terminal 2:  python collect_distillation_data.py
#
# =============================================================================

import os
import sys
import glob
import math
import weakref
import time
import random
import csv
import cv2
import logging
import numpy as np

from parameters import (
    TOWN, CAR_NAME,
    NUMBER_OF_VEHICLES, NUMBER_OF_PEDESTRIAN,
    VISUAL_DISPLAY, SEED,
    RESULTS_PATH,
)
from teacher import TeacherModel, ACTION_TO_CONTINUOUS

# CARLA egg import — works with both pip install and egg file
try:
    sys.path.append(glob.glob('./carla/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass  # carla installed via pip — no egg needed

import carla
import pygame

# =============================================================================
# COLLECTION PARAMETERS
# =============================================================================

COLLECTION_EPISODES  = 20      # total episodes to collect
FRAMES_PER_EPISODE   = 500     # max frames per episode
                               # 500 × 20 = 10,000 labelled frames total

# Navigation command sent to teacher every step
# 0=straight  1=left  2=right  3=follow lane
NAV_COMMAND          = 0

# CARLA synchronous fixed timestep
# 0.05s = 20 fps physics — teacher gets unlimited time between ticks
FIXED_DELTA_SECONDS  = 0.05

# Output paths
SAVE_DIR         = os.path.join(RESULTS_PATH, 'distillation_data')
SUMMARY_CSV_PATH = os.path.join(SAVE_DIR,     'collection_summary.csv')

# Spawn points per town — same as main.py
SPAWN_POINT          = {'Town07': 20, 'Town02': 30}
SPAWN_POINT_DEFAULT  = 12

# Route lengths per town — same as main.py
TOTAL_DISTANCE       = {'Town07': 750, 'Town02': 500}
TOTAL_DISTANCE_DEFAULT = 500

# Reward thresholds — must match main.py exactly
TARGET_SPEED = 22.0
MAX_SPEED    = 35.0
MIN_SPEED    = 15.0
MAX_DIST     =  3.0

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level   = logging.INFO,
    format  = '[%(asctime)s]  %(message)s',
    datefmt = '%H:%M:%S',
)
logger = logging.getLogger('DataCollector')

# =============================================================================
# CSV SCHEMA
# =============================================================================

EPISODE_CSV_HEADER = [
    # ── identifiers ───────────────────────────────────────────────────────────
    'frame_id',               # "frame_0042"
    'image_path',             # absolute path to saved PNG

    # ── INPUT columns (student receives at inference time) ────────────────────
    'velocity_kmh',           # speed in km/h
    'dist_from_center',       # metres from lane centre
    'angle_rad',              # heading error in radians
    'steering_angle',         # autopilot wheel angle [-1, +1]
    'throttle',               # autopilot throttle [0, 1]
    'nav_command',            # 0=straight 1=left 2=right 3=follow

    # ── LABEL columns (imitation learning target) ─────────────────────────────
    'teacher_action_label',   # e.g. "steer_left_slight"
    'teacher_steer',          # continuous steer  ∈ [-1, 1]
    'teacher_throttle_raw',   # continuous throttle_raw  ∈ [-1, 1]
    'teacher_reason',         # LLM justification sentence

    # ── AUTOPILOT columns (what actually drove the car) ───────────────────────
    'autopilot_steer',        # actual steer applied by autopilot
    'autopilot_throttle',     # actual throttle applied by autopilot

    # ── FEEDBACK columns (analysis only, not used in training) ────────────────
    'reward',                 # CARLA reward (same formula as main.py)
    'done',                   # 1 if episode ended this frame
    'teacher_agree',          # 1 if teacher label matches autopilot direction
    'step_time_s',            # wall-clock time for this step
]

SUMMARY_CSV_HEADER = [
    'episode',
    'total_frames',
    'total_reward',
    'avg_reward_per_frame',
    'waypoints_covered',
    'avg_lane_deviation_m',
    'teacher_agreement_pct',  # % frames where teacher agreed with autopilot
    'wall_time_s',
    'termination_reason',
    'episode_csv_path',
]

# =============================================================================
# CSV UTILITIES
# =============================================================================

def ensure_summary_csv():
    os.makedirs(SAVE_DIR, exist_ok=True)
    if not os.path.exists(SUMMARY_CSV_PATH):
        with open(SUMMARY_CSV_PATH, 'w', newline='') as f:
            csv.writer(f).writerow(SUMMARY_CSV_HEADER)


def write_summary_row(row: dict):
    with open(SUMMARY_CSV_PATH, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=SUMMARY_CSV_HEADER).writerow(row)


def init_episode_csv(path: str):
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerow(EPISODE_CSV_HEADER)


def append_frame_row(path: str, row: list):
    with open(path, 'a', newline='') as f:
        csv.writer(f).writerow(row)

# =============================================================================
# SENSOR CLASSES  (identical spec to main.py)
# =============================================================================

class SemanticCamera:
    """
    Front-facing semantic-segmentation camera.
    160×80 pixels, FoV 125°, attached at (x=2.4, z=1.5, pitch=-10).
    Identical spec to main.py CameraSensor.
    """

    def __init__(self, vehicle, world):
        self.latest_frame = []
        bp = world.get_blueprint_library().find(
            'sensor.camera.semantic_segmentation')
        bp.set_attribute('image_size_x', '160')
        bp.set_attribute('image_size_y', '80')
        bp.set_attribute('fov',          '125')
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=2.4, z=1.5),
                carla.Rotation(pitch=-10)),
            attach_to=vehicle)
        weak = weakref.ref(self)
        self.sensor.listen(lambda img: SemanticCamera._on_image(weak, img))

    @staticmethod
    def _on_image(weak, image):
        self = weak()
        if not self:
            return
        image.convert(carla.ColorConverter.CityScapesPalette)
        raw   = np.frombuffer(image.raw_data, dtype=np.uint8)
        frame = raw.reshape((image.width, image.height, 4))[:, :, :3]
        self.latest_frame.append(frame)

    def get_frame(self) -> np.ndarray:
        deadline = time.time() + 2.0
        while not self.latest_frame:
            if time.time() > deadline:
                logger.warning("Camera timeout — returning blank frame")
                return np.zeros((80, 160, 3), dtype=np.uint8)
            time.sleep(0.001)
        return self.latest_frame.pop(-1)

    def destroy(self):
        if self.sensor:
            self.sensor.destroy()
            self.sensor = None


class CollisionSensor:
    """Collision sensor — identical spec to main.py."""

    def __init__(self, vehicle, world):
        self.collision_data = []
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(carla.Location(x=1.3, z=0.5)),
            attach_to=vehicle)
        weak = weakref.ref(self)
        self.sensor.listen(lambda evt: CollisionSensor._on_collision(weak, evt))

    @staticmethod
    def _on_collision(weak, event):
        self = weak()
        if not self:
            return
        imp = event.normal_impulse
        self.collision_data.append(
            math.sqrt(imp.x**2 + imp.y**2 + imp.z**2))

    def destroy(self):
        if self.sensor:
            self.sensor.destroy()
            self.sensor = None


class DisplayCamera:
    """
    Optional third-person pygame display.
    Only created when VISUAL_DISPLAY = True in parameters.py.
    """

    def __init__(self, vehicle, world):
        pygame.init()
        self.display = pygame.display.set_mode(
            (600, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption('SmartDrive BTP — Data Collection')
        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', '600')
        bp.set_attribute('image_size_y', '600')
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=-4.0, z=2.0),
                carla.Rotation(pitch=-12.0)),
            attach_to=vehicle)
        weak = weakref.ref(self)
        self.sensor.listen(lambda img: DisplayCamera._on_image(weak, img))

    @staticmethod
    def _on_image(weak, image):
        self = weak()
        if not self:
            return
        arr  = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr  = arr.reshape((image.width, image.height, 4))[:, :, :3][:, :, ::-1]
        surf = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
        self.display.blit(surf, (0, 0))
        pygame.display.flip()

    def destroy(self):
        if self.sensor:
            self.sensor.destroy()
            self.sensor = None

# =============================================================================
# GEOMETRY HELPERS  (identical to main.py)
# =============================================================================

def vec3(v):
    if isinstance(v, (carla.Location, carla.Vector3D)):
        return np.array([v.x, v.y, v.z])
    if isinstance(v, carla.Rotation):
        return np.array([v.pitch, v.yaw, v.roll])
    return np.array(v)


def angle_diff(v0, v1):
    a = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
    if   a >  np.pi: a -= 2 * np.pi
    elif a <= -np.pi: a += 2 * np.pi
    return a


def dist_to_line(A, B, p):
    num   = np.linalg.norm(np.cross(B - A, A - p))
    denom = np.linalg.norm(B - A)
    return np.linalg.norm(p - A) if np.isclose(denom, 0) else num / denom

# =============================================================================
# WAYPOINT ROUTE BUILDER  (mirrors main.py exactly)
# =============================================================================

def build_route(world_map, start_location, town):
    total = TOTAL_DISTANCE.get(town, TOTAL_DISTANCE_DEFAULT)
    wp    = world_map.get_waypoint(
        start_location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving)
    route = [wp]
    for x in range(total):
        if town == 'Town07':
            nwp = wp.next(1.0)[0] if x < 650 else wp.next(1.0)[-1]
        elif town == 'Town02':
            nwp = wp.next(1.0)[-1] if x > 100 else wp.next(1.0)[0]
        else:
            nwp = wp.next(1.0)[-1] if x < 300 else wp.next(1.0)[0]
        route.append(nwp)
        wp = nwp
    return route

# =============================================================================
# REWARD FUNCTION  (identical to main.py — do NOT modify)
# =============================================================================

def compute_reward(velocity, dist_from_center, angle,
                   collision_history, episode_start_time):
    """Returns (reward, done, reason) — same logic as main.py."""
    if len(collision_history) != 0:
        return -10.0, True,  'collision'
    if dist_from_center > MAX_DIST:
        return -10.0, True,  'off_lane'
    if episode_start_time + 10 < time.time() and velocity < 1.0:
        return -10.0, True,  'stalled'
    if velocity > MAX_SPEED:
        return -10.0, True,  'overspeed'

    centering = max(1.0 - dist_from_center / MAX_DIST,   0.0)
    ang_fac   = max(1.0 - abs(angle) / np.deg2rad(20),   0.0)

    if velocity < MIN_SPEED:
        reward = (velocity / MIN_SPEED) * centering * ang_fac
    elif velocity > TARGET_SPEED:
        reward = (1.0 - (velocity - TARGET_SPEED) /
                  (MAX_SPEED - TARGET_SPEED)) * centering * ang_fac
    else:
        reward = 1.0 * centering * ang_fac

    return reward, False, 'running'

# =============================================================================
# TEACHER-AUTOPILOT AGREEMENT CHECK
# =============================================================================

def check_agreement(teacher_label: str, autopilot_steer: float) -> int:
    """
    Check if teacher label agrees with autopilot direction.
    Returns 1 if agreed, 0 if disagreed.

    Used for BTP analysis: how often does Qwen's reasoning
    match what the optimal autopilot actually does?
    """
    if teacher_label in ('go_straight', 'accelerate', 'decelerate', 'brake'):
        # Non-directional actions — agree if autopilot is also roughly straight
        return 1 if abs(autopilot_steer) < 0.1 else 0
    elif 'left' in teacher_label:
        return 1 if autopilot_steer < -0.05 else 0
    elif 'right' in teacher_label:
        return 1 if autopilot_steer >  0.05 else 0
    return 0

# =============================================================================
# NPC SPAWNERS
# =============================================================================

def spawn_npc_vehicles(client, world, bp_lib):
    ids = []
    spawn_points = world.get_map().get_spawn_points()
    for _ in range(NUMBER_OF_VEHICLES):
        bp = random.choice(bp_lib.filter('vehicle'))
        sp = random.choice(spawn_points)
        v  = world.try_spawn_actor(bp, sp)
        if v is not None:
            v.set_autopilot(True)
            ids.append(v.id)
    logger.info(f"NPC vehicles: {len(ids)}")
    return ids


def spawn_pedestrians(client, world, bp_lib):
    ids = []
    for _ in range(NUMBER_OF_PEDESTRIAN):
        sp  = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        sp.location = loc
        w_bp = random.choice(bp_lib.filter('walker.pedestrian.*'))
        c_bp = bp_lib.find('controller.ai.walker')
        if w_bp.has_attribute('is_invincible'):
            w_bp.set_attribute('is_invincible', 'false')
        if w_bp.has_attribute('speed'):
            w_bp.set_attribute(
                'speed', w_bp.get_attribute('speed').recommended_values[1])
        walker = world.try_spawn_actor(w_bp, sp)
        if walker is None:
            continue
        ctrl = world.spawn_actor(c_bp, carla.Transform(), walker)
        ids.extend([ctrl.id, walker.id])
    actors = world.get_actors(ids)
    for i in range(0, len(ids), 2):
        actors[i].start()
        actors[i].go_to_location(world.get_random_location_from_navigation())
    logger.info(f"Pedestrians: {len(ids)//2}")
    return ids

# =============================================================================
# SYNCHRONOUS MODE CONTEXT MANAGER
# =============================================================================

class SynchronousMode:
    """
    Switches CARLA to synchronous fixed-timestep mode.

    In synchronous mode the simulator does NOT advance until
    world.tick() is called. This gives Qwen-VL unlimited time
    to label each frame without the car moving during inference.

    Restores original settings on exit.
    """

    def __init__(self, world, delta=FIXED_DELTA_SECONDS):
        self.world = world
        self.delta = delta
        self._orig = None

    def __enter__(self):
        self._orig = self.world.get_settings()
        self.world.apply_settings(carla.WorldSettings(
            synchronous_mode    = True,
            fixed_delta_seconds = self.delta,
            no_rendering_mode   = False,
        ))
        logger.info(
            f"CARLA → synchronous mode  "
            f"(Δt={self.delta}s | {1/self.delta:.0f} fps)")
        return self

    def __exit__(self, *_):
        if self._orig:
            self.world.apply_settings(self._orig)
        logger.info("CARLA → restored async mode")

# =============================================================================
# MAIN COLLECTION LOOP
# =============================================================================

def collect_distillation_data():
    """
    Option C data collection — Autopilot Drives, Teacher Labels.

    Per episode:
      spawn vehicle
      enable autopilot          ← CARLA drives smoothly
      tick loop:
        world.tick()            ← advance simulator 0.05s
        get camera frame        ← semantic-seg 160×80
        read vehicle state      ← velocity, dist, angle, steer, throttle
        read autopilot control  ← what autopilot actually did
        advance waypoint index  ← identical to main.py
        Qwen-VL inference       ← IMAGE + STATE → label (NOT applied to car)
        compute reward          ← identical to main.py
        save PNG                ← raw frame for student input
        write CSV row           ← state + teacher_label + autopilot_action
      destroy vehicle + sensors
      write episode summary
    """

    # ── reproducibility ───────────────────────────────────────────────────────
    np.random.seed(SEED)
    random.seed(SEED)

    # ── output directories ────────────────────────────────────────────────────
    os.makedirs(SAVE_DIR, exist_ok=True)
    ensure_summary_csv()

    # ── CARLA connection ──────────────────────────────────────────────────────
    logger.info("Connecting to CARLA…")
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        world  = client.load_world(TOWN)
        world.set_weather(carla.WeatherParameters.CloudyNoon)
        logger.info(f"Connected ✓  town={TOWN}")
    except Exception as e:
        logger.error(f"CARLA connection failed: {e}")
        sys.exit(1)

    bp_lib    = world.get_blueprint_library()
    world_map = world.get_map()

    # ── load Qwen-VL teacher ──────────────────────────────────────────────────
    logger.info("Loading Qwen-VL teacher…")
    teacher = TeacherModel()
    logger.info("Teacher ready ✓")

    # ── spawn persistent NPCs + pedestrians ───────────────────────────────────
    npc_ids    = spawn_npc_vehicles(client, world, bp_lib)
    walker_ids = spawn_pedestrians(client, world, bp_lib)

    # ── synchronous mode ──────────────────────────────────────────────────────
    with SynchronousMode(world):

        for episode in range(1, COLLECTION_EPISODES + 1):

            logger.info(f"\n{'─'*60}")
            logger.info(f"  EPISODE  {episode:>3} / {COLLECTION_EPISODES}")
            logger.info(f"{'─'*60}")

            # ── episode paths ─────────────────────────────────────────────────
            ep_img_dir  = os.path.join(SAVE_DIR, f'Episode_{episode:02d}')
            ep_csv_path = os.path.join(SAVE_DIR, f'Episode_{episode:02d}_data.csv')
            os.makedirs(ep_img_dir, exist_ok=True)
            init_episode_csv(ep_csv_path)

            # ── spawn ego vehicle ─────────────────────────────────────────────
            veh_bp = bp_lib.filter(CAR_NAME)[0]
            if veh_bp.has_attribute('color'):
                veh_bp.set_attribute(
                    'color',
                    random.choice(
                        veh_bp.get_attribute('color').recommended_values))

            sp_idx    = SPAWN_POINT.get(TOWN, SPAWN_POINT_DEFAULT)
            transform = world_map.get_spawn_points()[sp_idx]
            vehicle   = world.try_spawn_actor(veh_bp, transform)

            if vehicle is None:
                logger.warning(f"Episode {episode}: spawn failed — skipping")
                continue

            # ── attach sensors ────────────────────────────────────────────────
            front_cam = SemanticCamera(vehicle, world)
            col_sens  = CollisionSensor(vehicle, world)
            disp_cam  = DisplayCamera(vehicle, world) if VISUAL_DISPLAY else None

            # ── ENABLE AUTOPILOT ──────────────────────────────────────────────
            # Autopilot drives the vehicle for the entire episode.
            # Teacher only observes and labels — never controls the car.
            vehicle.set_autopilot(True)

            # initialisation tick
            world.tick()

            # ── build waypoint route (for reward + waypoint tracking) ─────────
            route = build_route(world_map, vehicle.get_location(), TOWN)

            # ── episode state ─────────────────────────────────────────────────
            wp_index           = 0
            lane_dev_accum     = 0.0
            total_reward       = 0.0
            agree_count        = 0
            ep_start_time      = time.time()
            termination_reason = 'max_frames'

            # ══════════════════════════════════════════════════════════════════
            # FRAME LOOP
            # ══════════════════════════════════════════════════════════════════
            for frame_idx in range(FRAMES_PER_EPISODE):

                step_t0 = time.time()

                # ── STEP 1: TICK ──────────────────────────────────────────────
                # Advances simulator by FIXED_DELTA_SECONDS.
                # Autopilot has already submitted its control command.
                # All sensor callbacks fire here.
                world.tick()

                # ── STEP 2: GET CAMERA FRAME ──────────────────────────────────
                # Shape: (80, 160, 3)  dtype: uint8  channels: BGR
                image_array = front_cam.get_frame()

                # ── STEP 3: READ VEHICLE STATE ────────────────────────────────
                vel_vec  = vehicle.get_velocity()
                velocity = np.sqrt(
                    vel_vec.x**2 + vel_vec.y**2 + vel_vec.z**2) * 3.6
                location = vehicle.get_location()

                # ── STEP 4: READ AUTOPILOT CONTROL ────────────────────────────
                # Read what the autopilot actually did this tick.
                # Saved to CSV for comparison with teacher labels.
                ctrl              = vehicle.get_control()
                autopilot_steer   = float(ctrl.steer)
                autopilot_throttle = float(ctrl.throttle)

                # ── STEP 5: ADVANCE WAYPOINT INDEX ────────────────────────────
                idx = wp_index
                for _ in range(len(route)):
                    nwp = route[(idx + 1) % len(route)]
                    dot = np.dot(
                        vec3(nwp.transform.get_forward_vector())[:2],
                        vec3(location - nwp.transform.location)[:2])
                    if dot > 0.0:
                        idx += 1
                    else:
                        break
                wp_index = idx

                cur_wp  = route[wp_index       % len(route)]
                next_wp = route[(wp_index + 1) % len(route)]

                dist_from_center = dist_to_line(
                    vec3(cur_wp.transform.location),
                    vec3(next_wp.transform.location),
                    vec3(location))
                lane_dev_accum  += dist_from_center

                fwd_vel = vec3(vehicle.get_velocity())
                wp_fwd  = vec3(cur_wp.transform.rotation.get_forward_vector())
                ang     = angle_diff(fwd_vel, wp_fwd)

                # ── STEP 6: GREEN LIGHT OVERRIDE ──────────────────────────────
                if vehicle.is_at_traffic_light():
                    tl = vehicle.get_traffic_light()
                    if tl.get_state() == carla.TrafficLightState.Red:
                        tl.set_state(carla.TrafficLightState.Green)

                # ── STEP 7: QWEN-VL TEACHER LABELLING ────────────────────────
                # Simulator is FROZEN here while Qwen-VL thinks (~10s).
                # Teacher sees the frame and outputs what the CORRECT
                # action should be. This is the label — NOT applied to car.
                collision_flag = len(col_sens.collision_data) > 0
                try:
                    action_label, (t_steer, t_throttle_raw), reason = \
                        teacher.get_action(
                            image_obs            = image_array,
                            velocity             = velocity,
                            distance_from_center = dist_from_center,
                            angle                = ang,
                            steering_angle       = autopilot_steer,
                            throttle             = autopilot_throttle,
                            nav_command          = NAV_COMMAND,
                            collision_occurred   = collision_flag,
                        )
                except Exception as e:
                    logger.warning(
                        f"Ep{episode} frame{frame_idx}: teacher error: {e}")
                    action_label   = 'go_straight'
                    t_steer        = 0.0
                    t_throttle_raw = 0.6
                    reason         = 'teacher_error_fallback'

                # ── STEP 8: CHECK TEACHER-AUTOPILOT AGREEMENT ─────────────────
                # Did teacher's label match what autopilot actually did?
                agreed = check_agreement(action_label, autopilot_steer)
                agree_count += agreed

                # ── STEP 9: COMPUTE REWARD ────────────────────────────────────
                # Autopilot drives well so reward is usually high.
                # Frames with reward < 0 are filtered during imitation learning.
                reward, done, term_reason = compute_reward(
                    velocity           = velocity,
                    dist_from_center   = dist_from_center,
                    angle              = ang,
                    collision_history  = col_sens.collision_data,
                    episode_start_time = ep_start_time,
                )
                total_reward += reward
                if done:
                    termination_reason = term_reason

                # ── STEP 10: SAVE FRAME AS PNG ────────────────────────────────
                frame_name = f"frame_{frame_idx:04d}.png"
                frame_path = os.path.join(ep_img_dir, frame_name)
                cv2.imwrite(frame_path, image_array)

                # ── STEP 11: WRITE CSV ROW ────────────────────────────────────
                step_time = time.time() - step_t0
                row = [
                    # identifiers
                    f"frame_{frame_idx:04d}",        # frame_id
                    frame_path,                       # image_path

                    # INPUT columns — student receives at inference
                    round(velocity,           3),     # velocity_kmh
                    round(dist_from_center,   4),     # dist_from_center
                    round(ang,                6),     # angle_rad
                    round(autopilot_steer,    4),     # steering_angle
                    round(autopilot_throttle, 4),     # throttle
                    NAV_COMMAND,                      # nav_command

                    # LABEL columns — imitation learning target
                    action_label,                     # teacher_action_label
                    round(t_steer,            4),     # teacher_steer
                    round(t_throttle_raw,     4),     # teacher_throttle_raw
                    reason[:150],                     # teacher_reason

                    # AUTOPILOT columns — what actually drove
                    round(autopilot_steer,    4),     # autopilot_steer
                    round(autopilot_throttle, 4),     # autopilot_throttle

                    # FEEDBACK columns
                    round(reward,             4),     # reward
                    int(done),                        # done
                    agreed,                           # teacher_agree
                    round(step_time,          4),     # step_time_s
                ]
                append_frame_row(ep_csv_path, row)

                # ── STEP 12: LOG PROGRESS ─────────────────────────────────────
                if frame_idx % 50 == 0 or done:
                    agree_pct = (agree_count / max(frame_idx + 1, 1)) * 100
                    logger.info(
                        f"  Ep{episode:02d} | f={frame_idx:04d} | "
                        f"vel={velocity:5.1f} | dist={dist_from_center:.2f}m | "
                        f"label={action_label:<22} | "
                        f"agree={agree_pct:.0f}% | "
                        f"r={reward:+.3f} | t={step_time:.1f}s")

                if done:
                    logger.info(f"  Episode ended: {term_reason}")
                    break

            # ══════════════════════════════════════════════════════════════════
            # END OF EPISODE
            # ══════════════════════════════════════════════════════════════════
            ep_time       = time.time() - ep_start_time
            n_frames      = frame_idx + 1
            avg_lane_dev  = lane_dev_accum / max(n_frames, 1)
            agree_pct     = (agree_count / max(n_frames, 1)) * 100

            logger.info(
                f"\n  Ep{episode:02d} complete | "
                f"frames={n_frames} | "
                f"reward={total_reward:.2f} | "
                f"lane_dev={avg_lane_dev:.3f}m | "
                f"teacher_agreement={agree_pct:.1f}% | "
                f"time={ep_time:.1f}s | "
                f"reason={termination_reason}")

            write_summary_row({
                'episode':               episode,
                'total_frames':          n_frames,
                'total_reward':          round(total_reward,  3),
                'avg_reward_per_frame':  round(total_reward / max(n_frames, 1), 4),
                'waypoints_covered':     wp_index,
                'avg_lane_deviation_m':  round(avg_lane_dev,  4),
                'teacher_agreement_pct': round(agree_pct,      1),
                'wall_time_s':           round(ep_time,         2),
                'termination_reason':    termination_reason,
                'episode_csv_path':      ep_csv_path,
            })

            # ── clean up ego vehicle and sensors ──────────────────────────────
            front_cam.destroy()
            col_sens.destroy()
            if disp_cam:
                disp_cam.destroy()
            vehicle.destroy()
            world.tick()

    # ── restore async mode (SynchronousMode.__exit__) ─────────────────────────

    # ── tear down NPCs and pedestrians ────────────────────────────────────────
    logger.info("Cleaning up NPCs and pedestrians…")
    client.apply_batch(
        [carla.command.DestroyActor(x) for x in npc_ids + walker_ids])

    logger.info("\n" + "=" * 60)
    logger.info("  DISTILLATION DATA COLLECTION COMPLETE")
    logger.info(f"  Episodes  : {COLLECTION_EPISODES}")
    logger.info(f"  Saved to  : {SAVE_DIR}")
    logger.info(f"  Summary   : {SUMMARY_CSV_PATH}")
    logger.info("=" * 60)

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    try:
        collect_distillation_data()
    except KeyboardInterrupt:
        logger.info("\nInterrupted — exiting cleanly.")
        sys.exit(0)
    finally:
        print("\nData collection terminated.")