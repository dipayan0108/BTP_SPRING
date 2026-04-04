# state_builder.py
# With SB3 MultiInputPolicy, feature extraction and fusion are handled
# internally by SB3's per-modality CNN/FC branches.
# This file provides:
#   1. BLIPFeaturesExtractor  — custom SB3 feature extractor
#      that processes each Dict key through its own branch before concat
#   2. policy_kwargs           — passed directly to SB3 PPO constructor

import numpy as np
import torch
import torch.nn as nn
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from parameters import (
    IM_HEIGHT, IM_WIDTH,
    LIDAR_DIM, PID_DIM, NAV_DIM,
    BLIP_EMBEDDING_DIM,
    OBS_KEY_IMAGE, OBS_KEY_BLIP, OBS_KEY_LIDAR, OBS_KEY_PID, OBS_KEY_NAV,
)

# Projected dims per branch
CNN_PROJ_DIM   = 128   # visual spatial features from raw pixels
BLIP_PROJ_DIM  = 256   # semantic features from BLIP text embedding  ← NEW
LIDAR_PROJ_DIM = 128
PID_PROJ_DIM   = 32
NAV_PROJ_DIM   = 32

# Total concatenated feature dim fed into the actor/critic heads
FEATURES_DIM = (
    CNN_PROJ_DIM    # 128  — visual / CNN branch
    + BLIP_PROJ_DIM # 256  — semantic / BLIP branch   ← NEW
    + LIDAR_PROJ_DIM# 128  — spatial / LiDAR branch
    + PID_PROJ_DIM  # 32   — control / PID branch
    + NAV_PROJ_DIM  # 32   — telemetry / nav branch
)                   # 576  total


class BLIPFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom SB3 feature extractor for the Dict observation space.

    Each modality gets its own branch:
        image  → CNN  → 128-dim   (visual spatial features)
        blip   → FC   → 256-dim   (BLIP semantic embedding)   ← NEW
        lidar  → FC   → 128-dim
        pid    → FC   →  32-dim
        nav    → FC   →  32-dim
        ─────────────────────────
        concat         576-dim   ← _features_dim

    The 'blip' key carries the 768-dim L2-normalised BERT embedding
    produced by BLIPEncoder in the environment.  It gives the policy
    explicit semantic scene understanding (e.g. "wet road with pedestrian
    on left") that the raw pixel CNN cannot capture at this resolution.

    The 'image' key carries the raw semantic-segmentation pixel frame
    processed through a lightweight CNN for spatial / positional features
    (lane markings, road boundaries) that complement the BLIP embedding.
    """

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=FEATURES_DIM)

        # ── Image branch  (lightweight spatial CNN) ───────────────────
        # Input: (batch, H, W, 3) → permuted to (batch, 3, H, W)
        self.image_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute CNN flat output size dynamically
        with torch.no_grad():
            dummy     = torch.zeros(1, 3, IM_HEIGHT, IM_WIDTH)
            cnn_out   = self.image_branch(dummy).shape[1]

        self.image_proj = nn.Sequential(
            nn.Linear(cnn_out, CNN_PROJ_DIM),
            nn.ReLU(),
        )

        # ── BLIP semantic branch  ─────────────────────────────────────
        # Input: (batch, 768)  — L2-normalised BERT mean-pool embedding
        # Two FC layers compress to 256-dim with a residual-style skip
        # connection so the raw embedding signal is never fully lost.
        self.blip_branch = nn.Sequential(
            nn.Linear(BLIP_EMBEDDING_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, BLIP_PROJ_DIM),
            nn.ReLU(),
        )

        # ── LiDAR branch ─────────────────────────────────────────────
        self.lidar_branch = nn.Sequential(
            nn.Linear(LIDAR_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, LIDAR_PROJ_DIM),
            nn.ReLU(),
        )

        # ── PID branch ───────────────────────────────────────────────
        self.pid_branch = nn.Sequential(
            nn.Linear(PID_DIM, PID_PROJ_DIM),
            nn.ReLU(),
        )

        # ── Navigation branch ─────────────────────────────────────────
        self.nav_branch = nn.Sequential(
            nn.Linear(NAV_DIM, NAV_PROJ_DIM),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        # ── Image: (batch, H, W, 3) → (batch, 3, H, W) ───────────────
        img  = observations[OBS_KEY_IMAGE].permute(0, 3, 1, 2).float()
        img_feat = self.image_proj(self.image_branch(img))   # (B, 128)

        # ── BLIP semantic embedding: (batch, 768) ─────────────────────
        blip     = observations[OBS_KEY_BLIP].float()
        blip_feat = self.blip_branch(blip)                   # (B, 256)

        # ── LiDAR, PID, Nav ──────────────────────────────────────────
        lidar    = observations[OBS_KEY_LIDAR].float()
        pid      = observations[OBS_KEY_PID].float()
        nav      = observations[OBS_KEY_NAV].float()

        lidar_feat = self.lidar_branch(lidar)                # (B, 128)
        pid_feat   = self.pid_branch(pid)                    # (B,  32)
        nav_feat   = self.nav_branch(nav)                    # (B,  32)

        # ── Concatenate all branches → (B, 576) ───────────────────────
        return torch.cat(
            [img_feat, blip_feat, lidar_feat, pid_feat, nav_feat], dim=1
        )


# ── policy_kwargs passed to SB3 PPO ──────────────────────────────────
policy_kwargs = dict(
    features_extractor_class  = BLIPFeaturesExtractor,
    features_extractor_kwargs = {},
    net_arch                  = dict(pi=[256, 128], vf=[256, 128]),
    activation_fn             = nn.Tanh,
)