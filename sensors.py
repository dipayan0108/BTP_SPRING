# sensors.py
# CARLA sensor classes adapted from SmartDrive.
# Includes: CameraSensor (semantic segmentation), CameraSensorEnv
#           (third-person RGB display), CollisionSensor, LiDARSensor,
#           PIDController.
#
# FIX (memory / actor leak):
#   The original code stored only a weakref to the vehicle parent and
#   never explicitly stopped the sensor's listen() before destroying it.
#   Python's GC collected the Python sensor object while the CARLA actor
#   kept running in the simulation, producing:
#       WARNING: sensor object went out of the scope but the sensor is
#       still alive in the simulation: Actor NNN (sensor.*)
#   Each episode leaked 3 actor slots (camera, collision, lidar).
#
#   Fix applied to ALL sensor classes:
#     1. Call self.sensor.stop() before destroy — halts the data callback
#        so no new frames arrive after cleanup begins.
#     2. Expose a destroy() method that stops then destroys the actor.
#        CarlaEnv._destroy_actors() calls this instead of using
#        client.apply_batch([DestroyActor(sensor)]).
#     3. Keep a strong reference (self._parent_ref) alongside the weakref
#        so the sensor object is not GC'd while the episode is running.
#     4. Callback guard: weakref check is kept so stale callbacks after
#        stop() are silently ignored.
#
# FIX (camera buffer):
#   CameraSensor._callback() caps front_camera buffer at MAX_BUFFER=2
#   frames to prevent unbounded memory growth at high frame rates.

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

    Call destroy() explicitly to stop the listener and remove the actor
    from the simulation before releasing this object.
    """

    MAX_BUFFER = 2   # cap ring buffer — prevents unbounded memory growth

    def __init__(self, vehicle):
        self.sensor_name  = 'sensor.camera.semantic_segmentation'
        # FIX: keep a strong reference so GC cannot collect this object
        # while the episode is running.
        self._vehicle     = vehicle
        self.front_camera = []

        world        = vehicle.get_world()
        self.sensor  = self._setup(world)
        weak_self    = weakref.ref(self)
        self.sensor.listen(
            lambda image: CameraSensor._callback(weak_self, image)
        )

    def _setup(self, world):
        bp = world.get_blueprint_library().find(self.sensor_name)
        bp.set_attribute('image_size_x', str(IM_WIDTH))
        bp.set_attribute('image_size_y', str(IM_HEIGHT))
        bp.set_attribute('fov', '125')
        return world.spawn_actor(
            bp,
            carla.Transform(carla.Location(x=2.4, z=1.5),
                            carla.Rotation(pitch=-10)),
            attach_to=self._vehicle,
        )

    def destroy(self):
        """
        Stop the data callback and destroy the CARLA actor.
        Must be called before releasing this object to avoid the
        'sensor object went out of scope' warning.
        """
        if self.sensor is not None and self.sensor.is_alive:
            self.sensor.stop()    # halt callback delivery first
            self.sensor.destroy()
        self.sensor = None
        self.front_camera.clear()

    @staticmethod
    def _callback(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(carla.ColorConverter.CityScapesPalette)
        buf = np.frombuffer(image.raw_data, dtype=np.uint8)
        buf = buf.reshape((image.height, image.width, 4))
        if len(self.front_camera) >= CameraSensor.MAX_BUFFER:
            self.front_camera.pop(0)
        self.front_camera.append(buf[:, :, :3])   # drop alpha -> (H, W, 3)


# ═══════════════════════════════════════════════════════════════════════ #
#  Third-person RGB Camera  (visual display only)                        #
# ═══════════════════════════════════════════════════════════════════════ #

class CameraSensorEnv:
    """
    Third-person view rendered to a pygame window (display only).
    Call destroy() before releasing.
    """

    def __init__(self, vehicle):
        pygame.init()
        self.display = pygame.display.set_mode(
            (600, 600), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("BLIP-FusePPO — CARLA Live View")
        self.sensor_name = 'sensor.camera.rgb'
        # FIX: strong reference
        self._vehicle    = vehicle
        self.surface     = None

        world       = vehicle.get_world()
        self.sensor = self._setup(world)
        weak_self   = weakref.ref(self)
        self.sensor.listen(
            lambda image: CameraSensorEnv._callback(weak_self, image)
        )

    def _setup(self, world):
        bp = world.get_blueprint_library().find(self.sensor_name)
        bp.set_attribute('image_size_x', '600')
        bp.set_attribute('image_size_y', '600')
        return world.spawn_actor(
            bp,
            carla.Transform(carla.Location(x=-4.0, z=2.0),
                            carla.Rotation(pitch=-12.0)),
            attach_to=self._vehicle,
        )

    def destroy(self):
        if self.sensor is not None and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()
        self.sensor = None

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
    """
    Records collision impulse magnitudes into collision_data list.
    Call destroy() before releasing.
    """

    def __init__(self, vehicle):
        self.sensor_name    = 'sensor.other.collision'
        # FIX: strong reference
        self._vehicle       = vehicle
        self.collision_data = []

        world       = vehicle.get_world()
        self.sensor = self._setup(world)
        weak_self   = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._callback(weak_self, event)
        )

    def _setup(self, world):
        bp = world.get_blueprint_library().find(self.sensor_name)
        return world.spawn_actor(
            bp,
            carla.Transform(carla.Location(x=1.3, z=0.5)),
            attach_to=self._vehicle,
        )

    def destroy(self):
        if self.sensor is not None and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()
        self.sensor = None
        self.collision_data.clear()

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
    Values are clipped to [0, 100] m then divided by 100 -> [0, 1].
    Call destroy() before releasing.
    """

    LIDAR_RANGE    = 100.0
    NUM_CHANNELS   = 1
    POINTS_PER_SEC = 100000
    ROTATION_FREQ  = 20

    def __init__(self, vehicle):
        # FIX: strong reference
        self._vehicle   = vehicle
        self.range_data = np.ones(180, dtype=np.float32)

        world       = vehicle.get_world()
        self.sensor = self._setup(world)
        weak_self   = weakref.ref(self)
        self.sensor.listen(
            lambda data: LiDARSensor._callback(weak_self, data)
        )

    def _setup(self, world):
        bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        bp.set_attribute('channels',           str(self.NUM_CHANNELS))
        bp.set_attribute('range',              str(self.LIDAR_RANGE))
        bp.set_attribute('points_per_second',  str(self.POINTS_PER_SEC))
        bp.set_attribute('rotation_frequency', str(self.ROTATION_FREQ))
        bp.set_attribute('upper_fov',  '0')
        bp.set_attribute('lower_fov', '-10')
        return world.spawn_actor(
            bp,
            carla.Transform(carla.Location(x=0.0, z=2.4)),
            attach_to=self._vehicle,
        )

    def destroy(self):
        if self.sensor is not None and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()
        self.sensor = None

    @staticmethod
    def _callback(weak_self, data):
        self = weak_self()
        if not self:
            return

        n = len(data)
        if n == 0:
            return

        xs = np.empty(n, dtype=np.float32)
        ys = np.empty(n, dtype=np.float32)

        for i, detection in enumerate(data):
            xs[i] = detection.x
            ys[i] = detection.y

        # Filter to front 180° arc (x > 0 = forward half)
        front = xs > 0
        xs = xs[front]
        ys = ys[front]

        if len(xs) == 0:
            self.range_data = np.ones(180, dtype=np.float32)
            return

        distances  = np.sqrt(xs ** 2 + ys ** 2)
        angles_deg = np.degrees(np.arctan2(ys, xs))
        buckets    = np.clip(
            ((angles_deg + 90.0) / 180.0 * 179).astype(int), 0, 179
        )

        range_vec = np.full(180, self.LIDAR_RANGE, dtype=np.float32)
        for i, b in enumerate(buckets):
            if distances[i] < range_vec[b]:
                range_vec[b] = distances[i]

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
        self.Kp           = Kp
        self.Ki           = Ki
        self.Kd           = Kd
        self.max_integral = max_integral
        self.max_output   = max_output
        self.reset()

    def reset(self):
        self.integral   = 0.0
        self.prev_error = 0.0

    def compute(self, error, dt=1.0):
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