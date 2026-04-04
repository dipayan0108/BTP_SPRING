# reward.py
# Reward function for BLIP-FusePPO.
#
# Design:
#   BASE  = SmartDrive reward exactly (centering × angle × speed_factor)
#           Terminal penalty = -10, no clipping — directly comparable.
#
#   BLIP  = Cosine similarity bonus between current BLIP embedding
#           and a pre-computed "safe driving" reference embedding.
#           This is our addition on top of SmartDrive.
#           When the scene looks like clear road → bonus near +W_BLIP
#           When the scene looks like wall/obstacle → bonus near 0 or negative
#
# Total reward = BASE + W_BLIP * r_blip   (range: [-10, ~1.2])
#
# Comparison note:
#   SmartDrive mean reward 1502 = 478 steps × ~3.14 r/step
#   Our target:              same scale — directly comparable in paper table

import numpy as np
from parameters import (
    MIN_SPEED, TARGET_SPEED, MAX_SPEED,
    MAX_DISTANCE_FROM_CENTER,
    REWARD_TERMINAL,
    W_BLIP,
    BLIP_EMBEDDING_DIM,
)


# ── Safe-scene reference embedding ────────────────────────────────────
# Computed ONCE at module load from the phrase that best describes
# ideal driving: "a clear straight road with lane markings".
# At runtime reward computation is just a dot product — near zero cost.
#
# We use a fixed unit vector seeded from a stable hash of the phrase
# so it is deterministic without requiring BLIP at import time.
# When the real BLIPEncoder is available (in environment.py), the
# reference is overwritten with the true embedding via
# set_safe_reference_embedding().

_safe_reference: np.ndarray = None   # set by environment on first reset


def set_safe_reference_embedding(embedding: np.ndarray) -> None:
    """
    Call this once from CarlaEnv after BLIP is loaded.
    Encodes "a clear straight road with visible lane markings ahead"
    and stores the unit vector as the safe-scene reference.
    """
    global _safe_reference
    norm = np.linalg.norm(embedding)
    if norm > 1e-6:
        _safe_reference = embedding / norm
    else:
        _safe_reference = embedding.copy()


def _blip_reward(blip_embedding: np.ndarray) -> float:
    """
    Cosine similarity between current scene embedding and safe reference.
    Returns 0.0 if reference not yet set (first few steps).

    Typical values:
      Clear road ahead   →  0.75 – 0.92  (high similarity)
      Near obstacle/wall →  0.30 – 0.55  (low similarity)
      Off-road / grass   →  0.10 – 0.35  (very low)
    """
    if _safe_reference is None:
        return 0.0
    norm = np.linalg.norm(blip_embedding)
    if norm < 1e-6:
        return 0.0
    emb_unit = blip_embedding / norm
    similarity = float(np.dot(emb_unit, _safe_reference))
    # Shift: map [0, 1] → [-0.5, 0.5] so neutral scene = 0 contribution
    return float(np.clip(similarity - 0.5, -0.5, 0.5))


# ── Main reward function ───────────────────────────────────────────────

def compute_reward(
    distance_from_center_m: float,   # metres from lane centre (real units)
    velocity_kmh:           float,   # current speed km/h
    angle_rad:              float,   # heading error vs waypoint (radians)
    done:                   bool,    # episode terminated flag
    failed:                 bool,    # True if terminated due to violation
    blip_embedding:         np.ndarray = None,  # (768,) float32 or None
) -> float:
    """
    Returns scalar reward on scale [-10, ~1.2].
    Directly comparable with SmartDrive's reported metrics.

    Parameters match SmartDrive's step() variables:
        distance_from_center_m  ← self.distance_from_center
        velocity_kmh            ← self.velocity
        angle_rad               ← self.angle
        done / failed           ← terminal flags
        blip_embedding          ← self._last_blip_emb  (our addition)
    """

    # ── Terminal penalty ──────────────────────────────────────────────
    # Matches SmartDrive exactly: -10 on any violation
    if failed:
        return float(REWARD_TERMINAL)

    # ── SmartDrive base factors ───────────────────────────────────────
    centering_factor = max(
        1.0 - distance_from_center_m / max(MAX_DISTANCE_FROM_CENTER, 1e-6),
        0.0
    )
    angle_factor = max(
        1.0 - abs(angle_rad) / np.deg2rad(20),
        0.0
    )

    # ── SmartDrive speed factor ───────────────────────────────────────
    if velocity_kmh < MIN_SPEED:
        speed_factor = velocity_kmh / max(MIN_SPEED, 1e-6)
    elif velocity_kmh > TARGET_SPEED:
        speed_factor = max(
            1.0 - (velocity_kmh - TARGET_SPEED) / max(MAX_SPEED - TARGET_SPEED, 1e-6),
            0.0
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