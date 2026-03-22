# state_builder.py
# FIX: Added BLIP branch to BLIPFeaturesExtractor.
# The BLIP embedding (768-dim) was computed in the environment every K steps
# but never passed to the policy. It is now a proper obs key (OBS_KEY_BLIP)
# processed through a dedicated FC branch before concatenation.

import numpy as np
import torch
import torch.nn as nn
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from parameters import (
    IM_HEIGHT, IM_WIDTH,
    LIDAR_DIM, PID_DIM, NAV_DIM,
    BLIP_EMBEDDING_DIM,
    OBS_KEY_IMAGE, OBS_KEY_LIDAR, OBS_KEY_PID, OBS_KEY_NAV, OBS_KEY_BLIP,
)

# Projected dims per branch
CNN_PROJ_DIM   = 256   # image CNN spatial features
BLIP_PROJ_DIM  = 256   # BLIP semantic embedding features
LIDAR_PROJ_DIM = 128
PID_PROJ_DIM   = 32
NAV_PROJ_DIM   = 32
FEATURES_DIM   = CNN_PROJ_DIM + BLIP_PROJ_DIM + LIDAR_PROJ_DIM + PID_PROJ_DIM + NAV_PROJ_DIM  # 704


class BLIPFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom SB3 feature extractor for the Dict observation space.

    Each modality gets its own branch:
        image  → CNN  → 256-dim   (spatial features)
        blip   → FC   → 256-dim   (semantic language features)
        lidar  → FC   → 128-dim
        pid    → FC   →  32-dim
        nav    → FC   →  32-dim
        ─────────────────────────
        concat         704-dim   ← _features_dim
    """

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=FEATURES_DIM)

        # ── Image branch  (lightweight CNN) ──────────────────────────
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
            nn.Linear(cnn_out_dim, CNN_PROJ_DIM),
            nn.ReLU(),
        )

        # ── BLIP branch (semantic embedding) ─────────────────────────
        # FIX: was missing entirely — BLIP embeddings were discarded
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
        # Image: (batch, H, W, 3) → (batch, 3, H, W)
        img   = observations[OBS_KEY_IMAGE].permute(0, 3, 1, 2).float()
        blip  = observations[OBS_KEY_BLIP].float()   # FIX: now used
        lidar = observations[OBS_KEY_LIDAR].float()
        pid   = observations[OBS_KEY_PID].float()
        nav   = observations[OBS_KEY_NAV].float()

        img_feat   = self.image_proj(self.image_branch(img))
        blip_feat  = self.blip_branch(blip)            # FIX: now processed
        lidar_feat = self.lidar_branch(lidar)
        pid_feat   = self.pid_branch(pid)
        nav_feat   = self.nav_branch(nav)

        return torch.cat([img_feat, blip_feat, lidar_feat, pid_feat, nav_feat], dim=1)


# ── policy_kwargs passed to SB3 PPO ──────────────────────────────────
policy_kwargs = dict(
    features_extractor_class  = BLIPFeaturesExtractor,
    features_extractor_kwargs = {},
    net_arch                  = dict(pi=[256, 128], vf=[256, 128]),
    activation_fn             = nn.Tanh,
)