# sensors.py
# CARLA sensor classes adapted from SmartDrive.
# Includes: CameraSensor (semantic segmentation), CameraSensorEnv
#           (third-person RGB display), CollisionSensor, LiDARSensor.
#
# FIX: CameraSensor._callback() now caps front_camera buffer at 2 frames.
# Previously it grew unboundedly — at 20 FPS with slow BLIP inference,
# frames accumulated and caused memory growth (~38KB per frame).

import math
import weakref
import numpy as np
import pygame
import carla

from parameters import IM_WIDTH, IM_HEIGHT, VISUAL_DISPLAY


# ═══════════════════════════════════════════════════════════════════════ #
#  Semantic Segmentation Camera  (primary observation camera)            #
# ═══════════════════════════════════════════════════════════════════════ #

class CameraSensor:
    """
    Front-facing semantic segmentation camera.
    Outputs CityScapes-palette RGB images of size (IM_HEIGHT, IM_WIDTH, 3).
    These are fed directly into BLIP for semantic embedding extraction.
    """

    # FIX: cap ring buffer to prevent unbounded memory growth
    MAX_BUFFER = 2

    def __init__(self, vehicle):
        self.sensor_name = 'sensor.camera.semantic_segmentation'
        self.parent       = vehicle
        self.front_camera = []
        world             = self.parent.get_world()
        self.sensor       = self._setup(world)
        weak_self         = weakref.ref(self)
        self.sensor.listen(
            lambda image: CameraSensor._callback(weak_self, image)
        )

    def _setup(self, world):
        bp = world.get_blueprint_library().find(self.sensor_name)
        bp.set_attribute('image_size_x', str(IM_WIDTH))
        bp.set_attribute('image_size_y', str(IM_HEIGHT))
        bp.set_attribute('fov', '125')
        sensor = world.spawn_actor(
            bp,
            carla.Transform(carla.Location(x=2.4, z=1.5),
                            carla.Rotation(pitch=-10)),
            attach_to=self.parent,
        )
        return sensor

    @staticmethod
    def _callback(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(carla.ColorConverter.CityScapesPalette)
        buf = np.frombuffer(image.raw_data, dtype=np.uint8)
        buf = buf.reshape((image.height, image.width, 4))
        # FIX: evict oldest frame when buffer is full
        if len(self.front_camera) >= CameraSensor.MAX_BUFFER:
            self.front_camera.pop(0)
        self.front_camera.append(buf[:, :, :3])   # drop alpha → (H, W, 3)


# ═══════════════════════════════════════════════════════════════════════ #
#  Third-person RGB Camera  (visual display only)                        #
# ═══════════════════════════════════════════════════════════════════════ #

class CameraSensorEnv:
    """Third-person view rendered to a pygame window (display only)."""

    def __init__(self, vehicle):
        pygame.init()
        self.display = pygame.display.set_mode(
            (600, 600), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("BLIP-FusePPO — CARLA Live View")
        self.sensor_name = 'sensor.camera.rgb'
        self.parent       = vehicle
        self.surface      = None
        world             = self.parent.get_world()
        self.sensor       = self._setup(world)
        weak_self         = weakref.ref(self)
        self.sensor.listen(
            lambda image: CameraSensorEnv._callback(weak_self, image)
        )

    def _setup(self, world):
        bp = world.get_blueprint_library().find(self.sensor_name)
        bp.set_attribute('image_size_x', '600')
        bp.set_attribute('image_size_y', '600')
        sensor = world.spawn_actor(
            bp,
            carla.Transform(carla.Location(x=-4.0, z=2.0),
                            carla.Rotation(pitch=-12.0)),
            attach_to=self.parent,
        )
        return sensor

    @staticmethod
    def _callback(weak_self, image):
        self = weak_self()
        if not self:
            return
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
        arr = arr[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
        self.display.blit(self.surface, (0, 0))
        pygame.display.flip()


# ═══════════════════════════════════════════════════════════════════════ #
#  Collision Sensor                                                       #
# ═══════════════════════════════════════════════════════════════════════ #

class CollisionSensor:
    """Records collision impulse magnitudes into collision_data list."""

    def __init__(self, vehicle):
        self.sensor_name   = 'sensor.other.collision'
        self.parent        = vehicle
        self.collision_data = []
        world              = self.parent.get_world()
        self.sensor        = self._setup(world)
        weak_self          = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._callback(weak_self, event)
        )

    def _setup(self, world):
        bp = world.get_blueprint_library().find(self.sensor_name)
        sensor = world.spawn_actor(
            bp,
            carla.Transform(carla.Location(x=1.3, z=0.5)),
            attach_to=self.parent,
        )
        return sensor

    @staticmethod
    def _callback(weak_self, event):
        self = weak_self()
        if not self:
            return
        imp = event.normal_impulse
        self.collision_data.append(
            math.sqrt(imp.x ** 2 + imp.y ** 2 + imp.z ** 2)
        )


# ═══════════════════════════════════════════════════════════════════════ #
#  LiDAR Sensor                                                           #
# ═══════════════════════════════════════════════════════════════════════ #

class LiDARSensor:
    """
    180-degree horizontal LiDAR.
    Stores the latest normalised range vector in self.range_data (180,).
    Values are clipped to [0, 100] m then divided by 100 → [0, 1].
    """

    LIDAR_RANGE   = 100.0   # metres  (clip ceiling)
    NUM_CHANNELS  = 1
    POINTS_PER_SEC = 100000
    ROTATION_FREQ = 20

    def __init__(self, vehicle):
        self.parent     = vehicle
        self.range_data = np.ones(180, dtype=np.float32)  # safe default
        world           = self.parent.get_world()
        self.sensor     = self._setup(world)
        weak_self       = weakref.ref(self)
        self.sensor.listen(
            lambda data: LiDARSensor._callback(weak_self, data)
        )

    def _setup(self, world):
        bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        bp.set_attribute('channels',          str(self.NUM_CHANNELS))
        bp.set_attribute('range',             str(self.LIDAR_RANGE))
        bp.set_attribute('points_per_second', str(self.POINTS_PER_SEC))
        bp.set_attribute('rotation_frequency', str(self.ROTATION_FREQ))
        bp.set_attribute('upper_fov',  '0')
        bp.set_attribute('lower_fov', '-10')
        # NOTE: 'horizontal_fov' is NOT supported in CARLA 0.9.13 and below.
        # The full 360° sweep is used and filtered to the front 180° arc
        # in _callback() by keeping only points where x > 0 (forward half).
        sensor = world.spawn_actor(
            bp,
            carla.Transform(carla.Location(x=0.0, z=2.4)),
            attach_to=self.parent,
        )
        return sensor

    @staticmethod
    def _callback(weak_self, data):
        self = weak_self()
        if not self:
            return

        # ── Parse point cloud ─────────────────────────────────────────
        # CARLA 0.9.x on Windows returns raw_data that is NOT a simple
        # float32 buffer (sizes are not 16-byte aligned). The reliable
        # cross-platform approach is to iterate the measurement object
        # directly using CARLA's own point accessor, then build numpy
        # arrays from the extracted x/y/z values.
        n = len(data)   # number of LidarDetection points
        if n == 0:
            return

        xs = np.empty(n, dtype=np.float32)
        ys = np.empty(n, dtype=np.float32)

        for i, detection in enumerate(data):
            # In CARLA 0.9.13, iterating a LidarMeasurement yields
            # carla.Location objects directly — no .point wrapper.
            xs[i] = detection.x
            ys[i] = detection.y

        # ── Filter to front 180° arc (x > 0 = forward half) ──────────
        front = xs > 0
        xs = xs[front]
        ys = ys[front]

        if len(xs) == 0:
            self.range_data = np.ones(180, dtype=np.float32)
            return

        distances = np.sqrt(xs ** 2 + ys ** 2)

        # Map angle to bucket 0–179
        # arctan2(y, x) gives -90° (right) to +90° (left); shift to 0–179
        angles_deg = np.degrees(np.arctan2(ys, xs))
        buckets = np.clip(
            ((angles_deg + 90.0) / 180.0 * 179).astype(int), 0, 179
        )

        range_vec = np.full(180, self.LIDAR_RANGE, dtype=np.float32)
        for i, b in enumerate(buckets):
            if distances[i] < range_vec[b]:
                range_vec[b] = distances[i]

        # Normalise to [0, 1]
        self.range_data = np.clip(range_vec, 0, self.LIDAR_RANGE) / self.LIDAR_RANGE


# ═══════════════════════════════════════════════════════════════════════ #
#  PID Controller  (lateral steering correction)                         #
# ═══════════════════════════════════════════════════════════════════════ #

class PIDController:
    """
    Classic PID for lateral lane-centre correction.
    The scalar output is used as an auxiliary state feature (not direct
    actuation) following the BLIP-FusePPO design.
    """

    def __init__(self, Kp=0.15, Ki=0.02, Kd=0.08,
                 max_integral=1.0, max_output=0.8):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_integral = max_integral
        self.max_output   = max_output
        self.reset()

    def reset(self):
        self.integral   = 0.0
        self.prev_error = 0.0

    def compute(self, error: float, dt: float = 1.0) -> float:
        self.integral += error * dt
        self.integral  = np.clip(self.integral,
                                 -self.max_integral, self.max_integral)
        derivative = (error - self.prev_error) / max(dt, 1e-6)
        output = (self.Kp * error
                  + self.Ki * self.integral
                  + self.Kd * derivative)
        output = np.clip(output, -self.max_output, self.max_output)
        self.prev_error = error
        return float(output)