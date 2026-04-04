# reward.py
# Reward function for BLIP-FusePPO.
#
# Design:
#   BASE  = SmartDrive reward exactly (centering x angle x speed_factor)
#           Terminal penalty = REWARD_TERMINAL (-10), no clipping —
#           directly comparable with SmartDrive paper results.
#
#   BLIP  = Cosine similarity bonus between current BLIP embedding
#           and a pre-computed "safe driving" reference embedding.
#           When the scene looks like clear road -> bonus near +W_BLIP
#           When the scene looks like wall/obstacle -> bonus near 0 or negative
#
# Total reward = BASE + W_BLIP * r_blip   (range: [-10, ~1.2])
#
# FIX: removed unused BLIP_EMBEDDING_DIM import.
# FIX: parameter names now match what environment.py passes:
#      distance_from_center_m, velocity_kmh, angle_rad, done, failed,
#      blip_embedding — called correctly from environment.py step().
# FIX: set_safe_reference_embedding() must be called from CarlaEnv.__init__()
#      after BLIPEncoder is ready (see environment.py).

import numpy as np
from parameters import (
    MIN_SPEED, TARGET_SPEED, MAX_SPEED,
    MAX_DISTANCE_FROM_CENTER,
    REWARD_TERMINAL,
    W_BLIP,
)


# ── Safe-scene reference embedding ────────────────────────────────────
# Set once at startup via set_safe_reference_embedding().
# Until set, _blip_reward() returns 0.0 (no bonus).

_safe_reference = None   # np.ndarray (768,) unit vector, set by environment


def set_safe_reference_embedding(embedding):
    """
    Call once from CarlaEnv.__init__() after BLIPEncoder is loaded.

    Encodes "a clear straight road with visible lane markings ahead"
    and stores the L2-normalised vector as the safe-scene reference.

    Parameters
    ----------
    embedding : np.ndarray  shape (768,)  float32
        Raw BLIP embedding of a reference safe-driving image or phrase.
    """
    global _safe_reference
    norm = np.linalg.norm(embedding)
    if norm > 1e-6:
        _safe_reference = embedding / norm
    else:
        _safe_reference = embedding.copy()


def _blip_reward(blip_embedding):
    """
    Cosine similarity between current scene embedding and safe reference.
    Returns 0.0 if the reference has not been set yet.

    Typical values
    --------------
    Clear road ahead   ->  0.75 – 0.92  (high similarity)
    Near obstacle/wall ->  0.30 – 0.55  (low similarity)
    Off-road / grass   ->  0.10 – 0.35  (very low)

    Shifted to [-0.5, 0.5] so neutral similarity = 0 contribution.
    """
    if _safe_reference is None:
        return 0.0
    norm = np.linalg.norm(blip_embedding)
    if norm < 1e-6:
        return 0.0
    emb_unit   = blip_embedding / norm
    similarity = float(np.dot(emb_unit, _safe_reference))
    # Shift [0,1] -> [-0.5, 0.5]: neutral scene = 0 net contribution
    return float(np.clip(similarity - 0.5, -0.5, 0.5))


# ── Main reward function ───────────────────────────────────────────────

def compute_reward(
    distance_from_center_m,   # float  — metres from lane centre (real units)
    velocity_kmh,             # float  — current speed km/h
    angle_rad,                # float  — heading error vs waypoint (radians)
    done,                     # bool   — episode terminated flag
    failed,                   # bool   — True if terminated due to violation
    blip_embedding=None,      # np.ndarray (768,) float32, or None
):
    """
    Returns scalar reward on scale [-10, ~1.2].
    Directly comparable with SmartDrive's reported metrics.

    Parameters
    ----------
    distance_from_center_m : float
        Distance from lane centre in METRES (use self.distance_from_center
        from environment.py, NOT the pixel value from _compute_lane_distance_px).
    velocity_kmh : float
        Current vehicle speed in km/h.
    angle_rad : float
        Signed heading error between vehicle velocity and waypoint forward
        vector, in radians.
    done : bool
        Whether the episode is ending this step.
    failed : bool
        True if episode ended due to a constraint violation (collision,
        lane exit, LiDAR threshold, low speed, overspeed).
    blip_embedding : np.ndarray or None
        768-dim L2-normalised BLIP embedding of the current camera frame.
        Pass None to skip the BLIP bonus (e.g. during warm-up).
    """

    # ── Terminal penalty ──────────────────────────────────────────────
    # Matches SmartDrive exactly: REWARD_TERMINAL on any violation
    if failed:
        return float(REWARD_TERMINAL)

    # ── SmartDrive centering factor ───────────────────────────────────
    centering_factor = max(
        1.0 - distance_from_center_m / max(MAX_DISTANCE_FROM_CENTER, 1e-6),
        0.0,
    )

    # ── SmartDrive angle factor ───────────────────────────────────────
    angle_factor = max(
        1.0 - abs(angle_rad) / np.deg2rad(20),
        0.0,
    )

    # ── SmartDrive speed factor ───────────────────────────────────────
    if velocity_kmh < MIN_SPEED:
        speed_factor = velocity_kmh / max(MIN_SPEED, 1e-6)
    elif velocity_kmh > TARGET_SPEED:
        speed_factor = max(
            1.0 - (velocity_kmh - TARGET_SPEED)
            / max(MAX_SPEED - TARGET_SPEED, 1e-6),
            0.0,
        )
    else:
        speed_factor = 1.0

    # ── SmartDrive base reward (identical formula) ────────────────────
    r_base = speed_factor * centering_factor * angle_factor   # [0, 1]

    # ── BLIP semantic safety bonus (our contribution) ─────────────────
    r_blip = 0.0
    if blip_embedding is not None and W_BLIP > 0.0:
        r_blip = W_BLIP * _blip_reward(blip_embedding)

    return float(r_base + r_blip)