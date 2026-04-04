# exploration_callback.py
# Variable exploration noise decay — FIX-C applied.
#
# SmartDrive decays sigma every 300 episodes across 1200 total = 4 decay steps.
# Our training budget is ~400 episodes (100k steps at current episode lengths).
# ACTION_STD_DECAY_FREQ is now 75 in parameters.py, giving 4 decay steps
# across our run: 0.40 → 0.35 → 0.30 → 0.25 → 0.20, reaching ~0.20 at ep 300.
#
# The requires_grad fix from the original version is preserved.

import math
import torch
from stable_baselines3.common.callbacks import BaseCallback

from parameters import (
    ACTION_STD_INIT,
    ACTION_STD_MIN,
    ACTION_STD_DECAY,
    ACTION_STD_DECAY_FREQ,
    BLIP_WARMUP_EPISODES,
)


class ExplorationDecayCallback(BaseCallback):
    """
    Decays policy action std every ACTION_STD_DECAY_FREQ episodes.
    Also logs BLIP warmup status to the console.
    """

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self._current_std   = ACTION_STD_INIT
        self._episode_count = 0

    def _on_training_start(self):
        self._set_action_std(self._current_std)
        if self.verbose > 0:
            print(f"[ExplorationDecay] Initial sigma_noise = {self._current_std}")
            print(f"[ExplorationDecay] BLIP bonus active after episode {BLIP_WARMUP_EPISODES}")
            print(f"[ExplorationDecay] Decay schedule: every {ACTION_STD_DECAY_FREQ} episodes")

    def _on_step(self):
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [{}])

        for done, info in zip(dones, infos):
            if done:
                self._episode_count += 1

                # Log BLIP activation milestone
                ep_count = info.get("episode_count", self._episode_count)
                if ep_count == BLIP_WARMUP_EPISODES and self.verbose > 0:
                    print(f"\n[ExplorationDecay] BLIP reward bonus NOW ACTIVE (ep {ep_count})\n")

                if (self._episode_count % ACTION_STD_DECAY_FREQ == 0
                        and self._episode_count > 0):
                    self._current_std = max(
                        self._current_std - ACTION_STD_DECAY,
                        ACTION_STD_MIN,
                    )
                    self._set_action_std(self._current_std)

                    if self.verbose > 0:
                        print(
                            f"\n[ExplorationDecay] "
                            f"Episode {self._episode_count}: "
                            f"sigma_noise -> {self._current_std:.4f}\n"
                        )
        return True

    def _set_action_std(self, std):
        policy = self.model.policy
        if not hasattr(policy, 'log_std'):
            return
        log_std_value = float(math.log(max(std, 1e-8)))
        with torch.no_grad():
            policy.log_std.requires_grad_(False)
            policy.log_std.fill_(log_std_value)
            policy.log_std.requires_grad_(True)