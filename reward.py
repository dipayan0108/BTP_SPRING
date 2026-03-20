# reward.py
# BLIP-FusePPO hybrid nonlinear reward function.
# R = w_lane*r_lane + w_lidar*r_lidar + w_speed*r_speed + w_center*r_center

import numpy as np
from parameters import (
    W_LANE, W_LIDAR, W_SPEED, W_CENTER,
    LIDAR_DMID, LIDAR_DLOW, LIDAR_DCRIT,
    LIDAR_B1, LIDAR_B2, LIDAR_B3, LIDAR_BONUS,
    LIDAR_BONUS_RANGES,
    LANE_NORM, CENTER_MAX_DIST, CENTER_K,
    TARGET_SPEED, REWARD_CLIP, REWARD_FAIL,
)


def compute_reward(
    distance_px: float,       # lateral distance from lane centre (pixels)
    min_lidar_m: float,       # minimum LiDAR reading (metres, un-normalised)
    speed_kmh: float,         # current vehicle speed (km/h)
    done: bool,               # episode terminated flag
) -> float:
    """
    Returns a scalar reward clipped to [-REWARD_CLIP, +REWARD_CLIP].
    Returns REWARD_FAIL on episode termination.
    """

    if done:
        return float(REWARD_FAIL)

    # ── 1. Lane keeping reward ────────────────────────────────────────
    r_lane = 1.0 - abs(distance_px) / LANE_NORM
    r_lane = np.clip(r_lane, -1.0, 1.0)

    # ── 2. LiDAR obstacle avoidance reward ───────────────────────────
    r_lidar = _lidar_reward(min_lidar_m)

    # ── 3. Speed matching reward (quadratic penalty) ──────────────────
    r_speed = -((speed_kmh - TARGET_SPEED) / max(TARGET_SPEED, 1e-6)) ** 2

    # ── 4. Centralisation penalty (stronger for large deviations) ─────
    r_center = -CENTER_K * (abs(distance_px) / max(CENTER_MAX_DIST, 1e-6)) ** 2

    # ── Weighted sum ──────────────────────────────────────────────────
    reward = (W_LANE   * r_lane
            + W_LIDAR  * r_lidar
            + W_SPEED  * r_speed
            + W_CENTER * r_center)

    return float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))


def _lidar_reward(dmin: float) -> float:
    """
    Piecewise LiDAR reward from BLIP-FusePPO Table I.

      dmin ∈ [dlow, dmid]   → linear penalty
      dmin < dcrit          → heavy penalty
      dmin ∈ bonus ranges   → positive bonus
      otherwise             → 0
    """
    # Check bonus ranges first
    for lo, hi in LIDAR_BONUS_RANGES:
        if lo <= dmin <= hi:
            return float(LIDAR_BONUS)

    if dmin < LIDAR_DCRIT:
        return float(-LIDAR_B2 + LIDAR_B3 * dmin)

    if LIDAR_DLOW <= dmin <= LIDAR_DMID:
        return float(-LIDAR_B1 * (LIDAR_DMID - dmin) / max(LIDAR_DMID - LIDAR_DLOW, 1e-6))

    return 0.0
