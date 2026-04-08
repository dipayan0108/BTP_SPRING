# csv_logging_callback.py
# One CSV row per training episode — lean and diagnostic.
#
# COLUMN DECISIONS:
#   KEPT:    episode, timestep, wall_time_s          — position in training
#   KEPT:    episode_reward, mean_reward_10ep        — core convergence signal
#   KEPT:    reward_std_10ep                         — policy stability
#   KEPT:    reward_per_step                         — quality per step (normalises ep length)
#   KEPT:    episode_length, distance_covered_m      — driving performance
#   KEPT:    lane_deviation_m                        — matches SmartDrive metric
#   KEPT:    sigma_noise                             — tracks exploration decay progress
#   KEPT:    done_reason                             — diagnose why episodes end
#   KEPT:    r_lane, r_lidar, r_speed, r_center      — reward term breakdown for debugging
#
#   CUT:     mean_reward_50ep, mean_reward_100ep     — computable from episode_reward column
#   CUT:     cumulative_reward                       — computable as cumsum(episode_reward)
#   CUT:     best_reward_ever                        — computable as cummax(episode_reward)
#   These 4 were bloat — any analysis tool (pandas, Excel) computes them in one line.
#
#   TENSORBOARD already logs: episode_reward, distance_covered, lane_deviation,
#   mean_reward_10ep via RewardLoggingCallback — CSV is the persistent offline record.

import csv
import math
import os
import time
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from parameters import LOG_PATH_TRAIN

CSV_FILENAME = "training_log.csv"

CSV_COLUMNS = [
    # ── Position ──────────────────────────────────────────────────────
    "episode",
    "timestep",
    "wall_time_s",
    # ── Core reward ───────────────────────────────────────────────────
    "episode_reward",
    "mean_reward_10ep",
    "reward_std_10ep",
    "reward_per_step",
    # ── Driving performance ───────────────────────────────────────────
    "episode_length",
    "distance_covered_m",
    "lane_deviation_m",
    "done_reason",           # collision | lane_exit | destination | timeout
    # ── Exploration ───────────────────────────────────────────────────
    "sigma_noise",           # current policy std — confirms decay is working
    # ── Reward term breakdown ─────────────────────────────────────────
    # r_base_mean = SmartDrive base (speed x centering x angle) in [0,1]
    # Compare directly with SmartDrive's per-step ~0.77 to confirm parity.
    "r_base_mean",
    "r_lane_mean",
    "r_lidar_mean",
    "r_speed_mean",
    "r_center_mean",
]


class CSVLoggingCallback(BaseCallback):
    """
    Writes one CSV row per training episode.

    Reward term breakdown requires the environment to expose per-step
    reward components via info dict. If not present, columns default to 0.
    """

    def __init__(self, log_dir: str = LOG_PATH_TRAIN, verbose: int = 0):
        super().__init__(verbose)
        self._log_dir           = log_dir
        self._csv_path          = os.path.join(log_dir, CSV_FILENAME)
        self._file              = None
        self._writer            = None
        self._episode           = 0
        self._episode_reward    = 0.0
        self._episode_length    = 0
        self._training_start    = None
        self._reward_history    = []

        # Per-episode reward term accumulators
        self._r_base_sum   = 0.0
        self._r_lane_sum   = 0.0
        self._r_lidar_sum  = 0.0
        self._r_speed_sum  = 0.0
        self._r_center_sum = 0.0
        self._r_steps      = 0

    def _on_training_start(self) -> None:
        os.makedirs(self._log_dir, exist_ok=True)
        self._training_start = time.time()
        file_exists = os.path.isfile(self._csv_path)
        self._file   = open(self._csv_path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_COLUMNS)
        if not file_exists or os.path.getsize(self._csv_path) == 0:
            self._writer.writeheader()
            self._file.flush()
        if self.verbose > 0:
            print(f"[CSVLogger] Writing to {self._csv_path}")

    def _on_training_end(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file   = None
            self._writer = None

    def _get_sigma(self) -> float:
        """Read current policy std from log_std parameter."""
        try:
            policy = self.model.policy
            if hasattr(policy, 'log_std'):
                return float(torch.exp(policy.log_std).mean().item())
        except Exception:
            pass
        return 0.0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [0.0])
        dones   = self.locals.get("dones",   [False])
        infos   = self.locals.get("infos",   [{}])

        for reward, done, info in zip(rewards, dones, infos):
            self._episode_reward += float(reward)
            self._episode_length += 1

            # Accumulate individual reward terms if environment exposes them
            self._r_base_sum   += float(info.get("r_base",   0.0))
            self._r_lane_sum   += float(info.get("r_lane",   0.0))
            self._r_lidar_sum  += float(info.get("r_lidar",  0.0))
            self._r_speed_sum  += float(info.get("r_speed",  0.0))
            self._r_center_sum += float(info.get("r_center", 0.0))
            self._r_steps      += 1

            if done:
                self._episode += 1
                ep_r = self._episode_reward
                self._reward_history.append(ep_r)

                hist    = self._reward_history
                mean_10 = float(np.mean(hist[-10:]))
                std_10  = float(np.std(hist[-10:])) if len(hist) >= 2 else 0.0
                rps     = ep_r / self._episode_length if self._episode_length > 0 else 0.0

                # Reward term means (0 if env doesn't expose them)
                n = max(self._r_steps, 1)
                r_base_mean   = round(self._r_base_sum   / n, 5)
                r_lane_mean   = round(self._r_lane_sum   / n, 5)
                r_lidar_mean  = round(self._r_lidar_sum  / n, 5)
                r_speed_mean  = round(self._r_speed_sum  / n, 5)
                r_center_mean = round(self._r_center_sum / n, 5)

                # done_reason: environment sets this in info if available
                done_reason = info.get("done_reason", "unknown")

                row = {
                    "episode":          self._episode,
                    "timestep":         self.num_timesteps,
                    "wall_time_s":      round(time.time() - self._training_start, 1),
                    "episode_reward":   round(ep_r, 4),
                    "mean_reward_10ep": round(mean_10, 4),
                    "reward_std_10ep":  round(std_10, 4),
                    "reward_per_step":  round(rps, 6),
                    "episode_length":   self._episode_length,
                    "distance_covered_m": round(float(info.get("distance_covered", 0)), 1),
                    "lane_deviation_m": round(float(info.get("center_lane_deviation", 0)), 6),
                    "done_reason":      done_reason,
                    "sigma_noise":      round(self._get_sigma(), 5),
                    "r_base_mean":      r_base_mean,
                    "r_lane_mean":      r_lane_mean,
                    "r_lidar_mean":     r_lidar_mean,
                    "r_speed_mean":     r_speed_mean,
                    "r_center_mean":    r_center_mean,
                }

                self._writer.writerow(row)
                self._file.flush()

                if self.verbose > 0:
                    print(
                        f"[CSVLogger] Ep {self._episode:4d} | "
                        f"R={ep_r:7.3f} | "
                        f"avg10={mean_10:7.3f} | "
                        f"std={std_10:.3f} | "
                        f"dist={info.get('distance_covered',0):5.0f}m | "
                        f"reason={done_reason} | "
                        f"σ={self._get_sigma():.3f}"
                    )

                # Reset episode accumulators
                self._episode_reward = 0.0
                self._episode_length = 0
                self._r_base_sum     = 0.0
                self._r_lane_sum     = 0.0
                self._r_lidar_sum    = 0.0
                self._r_speed_sum    = 0.0
                self._r_center_sum   = 0.0
                self._r_steps        = 0

        return True