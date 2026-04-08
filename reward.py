# reward.py
#
# FORMULA (per step, no clip):
#   r = speed_factor x centering_factor x angle_factor   [SmartDrive base, 0-1]
#     + W_LIDAR  x r_lidar                               [safety bonus, <=0.3]
#     + W_SPEED  x r_speed                               [extra gradient below target]
#
# LIDAR ZONES (continuous, no cliffs, no dead zones):
#   d < 2.0m          : critical penalty  (-5 → -1, linear)
#   2.0 - 3.0m        : warning penalty   (-1 → 0, linear)
#   3.0 - 5.0m        : safe zone 1 bonus (+1.0)  car-width clearance
#   5.0 - 7.0m        : neutral           (0)
#   7.0 - 10.0m       : safe zone 2 bonus (+0.6)  open road ahead
#   > 10m             : neutral           (0)
#
# PROPERTIES:
#   - Stopped car cannot earn positive reward (LiDAR gated on speed_factor)
#   - LiDAR bonus (max 0.3) always < base (max 1.0) — base dominates
#   - Terminal -100 / good step ~1.3 = 77x deterrence (vs SmartDrive 13x)
#   - Base alone beats SmartDrive/step: 1.0 > 0.77

import numpy as np
from parameters import (
    MIN_SPEED, TARGET_SPEED, MAX_SPEED,
    MAX_DISTANCE_FROM_CENTER,
    REWARD_TERMINAL,
    W_LIDAR, W_SPEED,
    LIDAR_DCRIT, LIDAR_DLOW,
    LIDAR_SAFE1_LO, LIDAR_SAFE1_HI,
    LIDAR_SAFE2_LO, LIDAR_SAFE2_HI,
    LIDAR_BONUS, LIDAR_BONUS2,
)


# ── SmartDrive base factors ────────────────────────────────────────────

def _speed_factor(v):
    if v < MIN_SPEED:
        return v / max(MIN_SPEED, 1e-6)
    if v > TARGET_SPEED:
        return max(1.0 - (v - TARGET_SPEED) / max(MAX_SPEED - TARGET_SPEED, 1e-6), 0.0)
    return 1.0


def _centering_factor(d):
    return max(1.0 - d / max(MAX_DISTANCE_FROM_CENTER, 1e-6), 0.0)


def _angle_factor(a):
    return max(1.0 - abs(a) / np.deg2rad(20), 0.0)


# ── LiDAR safety term (continuous, no cliffs) ─────────────────────────

def _r_lidar_raw(d):
    """
    Continuous piecewise LiDAR reward. No gaps or cliffs.

    Zones:
      d <  2.0m  : critical  — linear penalty -5+2.5d  (-5 at 0m, -1 at 2m)
      2.0-3.0m   : warning   — linear penalty -(3-d)    (-1 at 2m,  0 at 3m)
      3.0-5.0m   : safe 1    — bonus +1.0
      5.0-7.0m   : neutral   — 0
      7.0-10.0m  : safe 2    — bonus +0.6
      > 10m      : neutral   — 0
    """
    if d < LIDAR_DCRIT:                        # < 2.0m: critical
        return -5.0 + 2.5 * d
    if LIDAR_DCRIT <= d < LIDAR_DLOW:          # 2.0-3.0m: warning
        return -(LIDAR_DLOW - d)
    if LIDAR_SAFE1_LO <= d <= LIDAR_SAFE1_HI:  # 3.0-5.0m: safe zone 1
        return float(LIDAR_BONUS)
    if LIDAR_SAFE2_LO <= d <= LIDAR_SAFE2_HI:  # 7.0-10.0m: safe zone 2
        return float(LIDAR_BONUS2)
    return 0.0                                  # neutral


# ── Diagnostic terms for CSV logger ───────────────────────────────────

def reward_terms(
    distance_from_center_m=0.0,
    velocity_kmh=0.0,
    angle_rad=0.0,
    min_lidar_m=100.0,
    distance_px=0.0,
):
    sf = _speed_factor(velocity_kmh)
    cf = _centering_factor(distance_from_center_m)
    af = _angle_factor(angle_rad)
    r_base  = sf * cf * af
    lidar_raw = _r_lidar_raw(min_lidar_m)
    lidar_gated = lidar_raw if sf > 0.1 else 0.0
    r_lidar = W_LIDAR * float(np.clip(lidar_gated, -5.0, LIDAR_BONUS))
    r_speed = W_SPEED * max(-((velocity_kmh - TARGET_SPEED)**2)
                            / (TARGET_SPEED**2), -0.5) if sf < 1.0 else 0.0
    return {
        "r_base":   r_base,
        "r_lane":   r_base,
        "r_lidar":  float(r_lidar),
        "r_speed":  float(r_speed),
        "r_center": float(cf),
    }


# ── Main reward ────────────────────────────────────────────────────────

def compute_reward(
    distance_from_center_m,
    velocity_kmh,
    angle_rad,
    done,
    failed,
    distance_px=0.0,
    min_lidar_m=100.0,
):
    if failed:
        return float(REWARD_TERMINAL)

    sf = _speed_factor(velocity_kmh)
    cf = _centering_factor(distance_from_center_m)
    af = _angle_factor(angle_rad)
    r_base = sf * cf * af

    lidar_raw   = _r_lidar_raw(min_lidar_m)
    lidar_gated = lidar_raw if sf > 0.1 else 0.0
    r_lidar = W_LIDAR * float(np.clip(lidar_gated, -5.0, LIDAR_BONUS))

    r_speed = W_SPEED * max(-((velocity_kmh - TARGET_SPEED)**2)
                            / (TARGET_SPEED**2), -0.5) if sf < 1.0 else 0.0

    return float(r_base + r_lidar + r_speed)