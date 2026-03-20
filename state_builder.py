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
    OBS_KEY_IMAGE, OBS_KEY_LIDAR, OBS_KEY_PID, OBS_KEY_NAV,
)

# Projected dims per branch
BLIP_PROJ_DIM  = 256
LIDAR_PROJ_DIM = 128
PID_PROJ_DIM   = 32
NAV_PROJ_DIM   = 32
FEATURES_DIM   = BLIP_PROJ_DIM + LIDAR_PROJ_DIM + PID_PROJ_DIM + NAV_PROJ_DIM  # 448


class BLIPFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom SB3 feature extractor for the Dict observation space.

    Each modality gets its own branch:
        image  → CNN  → 256-dim
        lidar  → FC   → 128-dim
        pid    → FC   →  32-dim
        nav    → FC   →  32-dim
        ─────────────────────────
        concat         448-dim   ← _features_dim

    Note: the 'image' key holds a semantically-segmented RGB frame
    which is used as the visual input to BLIP (done in the environment).
    Here we process the same image through a lightweight CNN to extract
    spatial features in parallel with the BLIP semantic branch.
    The BLIP embedding itself is not a separate gym obs key — it is
    computed inside the environment and the image key is reused so that
    SB3 can work with a standard Dict space.
    """

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=FEATURES_DIM)

        # ── Image branch  (lightweight CNN) ──────────────────────────
        # Input: (batch, H, W, 3) → permuted to (batch, 3, H, W) for PyTorch
        self.image_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute CNN output size
        dummy = torch.zeros(1, 3, IM_HEIGHT, IM_WIDTH)
        cnn_out_dim = self.image_branch(dummy).shape[1]

        self.image_proj = nn.Sequential(
            nn.Linear(cnn_out_dim, BLIP_PROJ_DIM),
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
        # Image: (batch, H, W, 3) → (batch, 3, H, W)
        img   = observations[OBS_KEY_IMAGE].permute(0, 3, 1, 2).float()
        lidar = observations[OBS_KEY_LIDAR].float()
        pid   = observations[OBS_KEY_PID].float()
        nav   = observations[OBS_KEY_NAV].float()

        img_feat   = self.image_proj(self.image_branch(img))
        lidar_feat = self.lidar_branch(lidar)
        pid_feat   = self.pid_branch(pid)
        nav_feat   = self.nav_branch(nav)

        return torch.cat([img_feat, lidar_feat, pid_feat, nav_feat], dim=1)


# ── policy_kwargs passed to SB3 PPO ──────────────────────────────────
policy_kwargs = dict(
    features_extractor_class  = BLIPFeaturesExtractor,
    features_extractor_kwargs = {},
    net_arch                  = dict(pi=[256, 128], vf=[256, 128]),
    activation_fn             = nn.Tanh,
)