# reward.py
# Hybrid reward for SmartDrive + BLIP-FusePPO in CARLA.
#
# DESIGN (matches paper Section III-C exactly):
#   Rt = W_LANE   * r_lane
#      + W_LIDAR  * r_lidar
#      + W_SPEED  * r_speed
#      + W_CENTER * r_center
#
# FIX-1 (critical): BLIP removed from reward entirely.
#   The paper's key contribution is BLIP-in-STATE not BLIP-in-reward.
#   Prior work (VL-SAFE etc.) used VLMs for reward shaping — the paper
#   explicitly improves upon this by injecting semantics into observations.
#   Having BLIP in both state AND reward was double-counting and adding
#   a noisy cosine-similarity signal that fought convergence.
#
# FIX-7: LiDAR reward term now wired in (was defined in parameters.py
#   but never called from the reward function).
#
# FIX-speed: speed term uses paper's quadratic penalty -(v-vt)²/vt²
#   instead of SmartDrive's piecewise linear. Quadratic gives a smooth
#   gradient that always pushes toward vtarget from both sides.

import numpy as np
from parameters import (
    TARGET_SPEED,
    REWARD_TERMINAL,
    W_LANE, W_LIDAR, W_SPEED, W_CENTER,
    LANE_NORM, CENTER_MAX_DIST, CENTER_K, REWARD_CLIP,
    LIDAR_DMID, LIDAR_DLOW, LIDAR_DCRIT,
    LIDAR_B1, LIDAR_B2, LIDAR_B3,
    LIDAR_BONUS, LIDAR_BONUS_RANGES,
)


# ── Lane-keeping reward (paper Eq.14) ─────────────────────────────────

def _r_lane(distance_px):
    """
    Linear reward for lateral closeness to lane centre.
    r_lane = 1 - |delta_x| / d_lane
    Clipped to [0, 1].
    """
    return float(np.clip(1.0 - abs(distance_px) / max(LANE_NORM, 1e-6),
                         0.0, 1.0))


# ── LiDAR obstacle-avoidance reward (paper Eq.15) ─────────────────────

def _r_lidar(min_lidar_m):
    """
    Piecewise reward based on minimum LiDAR reading (metres).

    Zones:
      dmin in [dlow, dmid]         -> moderate penalty (approaching)
      dmin < dcrit                 -> heavy penalty (critically close)
      dmin in bonus_ranges         -> positive bonus (safe spacing)
      otherwise                   -> 0
    """
    dmin = min_lidar_m

    # Bonus zones first (safe clearance)
    for (lo, hi) in LIDAR_BONUS_RANGES:
        if lo <= dmin <= hi:
            return float(LIDAR_BONUS)

    # Critical zone (very close to obstacle)
    if dmin < LIDAR_DCRIT:
        return float(-LIDAR_B2 + LIDAR_B3 * dmin)

    # Approaching zone
    if LIDAR_DLOW <= dmin <= LIDAR_DMID:
        drange = max(LIDAR_DMID - LIDAR_DLOW, 1e-6)
        return float(-LIDAR_B1 * (LIDAR_DMID - dmin) / drange)

    return 0.0


# ── Speed reward (paper Eq.16 — quadratic) ────────────────────────────

def _r_speed(velocity_kmh):
    """
    Quadratic penalty for deviation from target speed.
    r_speed = -(v - vtarget)^2 / vtarget^2
    Range: (-inf, 0] — always non-positive, zero only at vtarget.
    Gives smooth gradient from both under-speed and over-speed.
    """
    vt = max(TARGET_SPEED, 1e-6)
    return float(-((velocity_kmh - vt) ** 2) / (vt ** 2))


# ── Centre penalty (paper Eq.17 — quadratic) ──────────────────────────

def _r_center(distance_px):
    """
    Quadratic penalty for large lateral offsets.
    r_center = -k * (|delta_x| / d_clip)^2
    """
    dclip = max(CENTER_MAX_DIST, 1e-6)
    return float(-CENTER_K * (abs(distance_px) / dclip) ** 2)


# ── Main reward function ───────────────────────────────────────────────

def reward_terms(
    distance_px: float = 0.0,
    min_lidar_m: float = 100.0,
    velocity_kmh: float = 0.0,
) -> dict:
    """
    Returns the individual unweighted reward components as a dict.
    Used by environment.py to populate info dict for CSV logging.
    Call this BEFORE compute_reward() so terms are always available.
    """
    return {
        "r_lane":   _r_lane(distance_px),
        "r_lidar":  _r_lidar(min_lidar_m),
        "r_speed":  _r_speed(velocity_kmh),
        "r_center": _r_center(distance_px),
    }


def compute_reward(
    distance_from_center_m,   # float  — metres from lane centre
    velocity_kmh,             # float  — current speed km/h
    angle_rad,                # float  — heading error vs waypoint (radians)
    done,                     # bool   — episode terminated flag
    failed,                   # bool   — True if terminated due to violation
    distance_px=0.0,          # float  — lateral offset in pixels (Hough)
    min_lidar_m=100.0,        # float  — minimum LiDAR reading in metres
):
    """
    Returns scalar reward.

    Terminal:  REWARD_TERMINAL (-10) on any constraint violation.
    Normal:    Weighted sum of 4 terms from paper Table I.

    Scale is directly comparable with SmartDrive's reported metrics
    because the base lane/speed/center logic maps to the same driving
    quality criteria (SmartDrive uses a multiplicative form; we use
    the paper's additive weighted form which is equivalent at optimum).

    Parameters
    ----------
    distance_from_center_m : float
        Metres from lane centre (from waypoint geometry, real units).
    velocity_kmh : float
        Current vehicle speed in km/h.
    angle_rad : float
        Signed heading error in radians.
    done : bool
        Episode is ending this step.
    failed : bool
        Ended due to constraint violation (collision, lane exit, etc.).
    distance_px : float
        Lateral offset in pixels from Hough lane detector (for r_lane,
        r_center terms which use pixel scale per paper Table I).
    min_lidar_m : float
        Minimum LiDAR range reading in metres (for r_lidar term).
    """

    # ── Terminal penalty ──────────────────────────────────────────────
    if failed:
        return float(REWARD_TERMINAL)

    # ── Four reward terms (paper Eq.13) ──────────────────────────────
    r_lane   = _r_lane(distance_px)
    r_lidar  = _r_lidar(min_lidar_m)
    r_speed  = _r_speed(velocity_kmh)
    r_center = _r_center(distance_px)

    rt = (W_LANE   * r_lane
        + W_LIDAR  * r_lidar
        + W_SPEED  * r_speed
        + W_CENTER * r_center)

    # ── Clip per paper Eq.19 ─────────────────────────────────────────
    # Note: REWARD_CLIP=2.0 (raised from paper's 1.0) so that the LiDAR
    # bonus (W_LIDAR * LIDAR_BONUS = 0.3 * 5 = 1.5) is not crushed to 1.0
    # alongside a positive r_lane term. The terminal penalty (-10) is NOT
    # clipped — it bypasses this line via the early-return above.
    rt = float(np.clip(rt, -REWARD_CLIP, REWARD_CLIP))

    return rt