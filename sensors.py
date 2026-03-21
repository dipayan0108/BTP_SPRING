# sensors.py
# CARLA sensor classes adapted from SmartDrive.
# Includes: CameraSensor (semantic segmentation), CameraSensorEnv
#           (third-person RGB display), CollisionSensor, LiDARSensor.

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

    def __init__(self, vehicle):
        self.sensor_name = 'sensor.camera.semantic_segmentation'
        self.parent       = vehicle
        self.front_camera = []          # ring buffer; pop(-1) to read latest
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
        bp.set_attribute('rotation_frequency',str(self.ROTATION_FREQ))
        bp.set_attribute('upper_fov',  '0')
        bp.set_attribute('lower_fov', '-10')
        # 180-degree horizontal sweep centred on the front
        bp.set_attribute('horizontal_fov', '180')
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
        # Each point: (x, y, z, intensity)
        points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
        if len(points) == 0:
            return
        # Euclidean distance per point
        distances = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        # Bin into 180 angular buckets  (0° = front, ±90° = sides)
        angles = np.degrees(np.arctan2(points[:, 1], points[:, 0])) + 90.0
        angles = np.clip(angles, 0, 179).astype(int)
        range_vec = np.full(180, self.LIDAR_RANGE, dtype=np.float32)
        for i, ang in enumerate(angles):
            if distances[i] < range_vec[ang]:
                range_vec[ang] = distances[i]
        # Clip & normalise to [0, 1]
        range_vec = np.clip(range_vec, 0, self.LIDAR_RANGE) / self.LIDAR_RANGE
        self.range_data = range_vec


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