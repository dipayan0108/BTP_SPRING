# csv_logging_callback.py
# Writes one CSV row per training episode with comprehensive reward metrics.
#
# Reward metrics tracked:
#   episode_reward     — absolute total reward for the episode (SUM)
#   mean_reward_10ep   — rolling mean over last 10 episodes
#   mean_reward_50ep   — rolling mean over last 50 episodes
#   mean_reward_100ep  — rolling mean over last 100 episodes
#   best_reward_ever   — highest single episode reward seen so far
#   reward_std_10ep    — std deviation over last 10 eps (consistency)
#   reward_per_step    — episode_reward / episode_length (quality per step)
#   cumulative_reward  — running total reward across all episodes

import csv
import os
import time
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from parameters import LOG_PATH_TRAIN

CSV_FILENAME = "training_log.csv"

CSV_COLUMNS = [
    "episode",
    "timestep",
    "wall_time_s",
    # Reward metrics
    "episode_reward",
    "mean_reward_10ep",
    "mean_reward_50ep",
    "mean_reward_100ep",
    "best_reward_ever",
    "reward_std_10ep",
    "reward_per_step",
    "cumulative_reward",
    # Episode metrics
    "episode_length",
    "distance_covered_m",
    "lane_deviation_m",

]


class CSVLoggingCallback(BaseCallback):
    """
    Writes one CSV row per training episode.
    Tracks both absolute and averaged reward metrics.
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
        self._best_reward       = float('-inf')
        self._cumulative_reward = 0.0

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

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [0.0])
        dones   = self.locals.get("dones",   [False])
        infos   = self.locals.get("infos",   [{}])

        for reward, done, info in zip(rewards, dones, infos):
            self._episode_reward += float(reward)
            self._episode_length += 1

            if done:
                self._episode          += 1
                ep_r                    = self._episode_reward
                self._cumulative_reward += ep_r
                self._reward_history.append(ep_r)
                self._best_reward = max(self._best_reward, ep_r)

                hist     = self._reward_history
                mean_10  = float(np.mean(hist[-10:]))
                mean_50  = float(np.mean(hist[-50:]))
                mean_100 = float(np.mean(hist[-100:]))
                std_10   = float(np.std(hist[-10:])) if len(hist) >= 2 else 0.0
                rps      = ep_r / self._episode_length if self._episode_length > 0 else 0.0

                row = {
                    "episode":           self._episode,
                    "timestep":          self.num_timesteps,
                    "wall_time_s":       round(time.time() - self._training_start, 2),
                    "episode_reward":    round(ep_r, 4),
                    "mean_reward_10ep":  round(mean_10, 4),
                    "mean_reward_50ep":  round(mean_50, 4),
                    "mean_reward_100ep": round(mean_100, 4),
                    "best_reward_ever":  round(self._best_reward, 4),
                    "reward_std_10ep":   round(std_10, 4),
                    "reward_per_step":   round(rps, 6),
                    "cumulative_reward": round(self._cumulative_reward, 4),
                    "episode_length":    self._episode_length,
                    "distance_covered_m": info.get("distance_covered", 0),
                    "lane_deviation_m":  round(info.get("center_lane_deviation", 0), 6),
                }

                self._writer.writerow(row)
                self._file.flush()

                if self.verbose > 0:
                    print(
                        f"[CSVLogger] Ep {self._episode:4d} | "
                        f"R={ep_r:7.2f} | "
                        f"avg10={mean_10:7.2f} | "
                        f"avg100={mean_100:7.2f} | "
                        f"best={self._best_reward:7.2f} | "
                        f"std={std_10:.3f} | "
                        f"R/step={rps:.4f}"
                    )

                self._episode_reward = 0.0
                self._episode_length = 0

        return True