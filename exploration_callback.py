# exploration_callback.py
# Variable exploration noise decay — SmartDrive strategy adapted for SB3.
#
# FIX-5: Episode counter now synced from env info dict instead of an
#   internal counter that reset to 0 on checkpoint resume.
#   Previously: self._episode_count started at 0 every run, so sigma
#   always started at ACTION_STD_INIT=0.4 even when loading from ep 200.
#   Now: reads episode_count from info dict that environment provides,
#   which persists across checkpoint loads via env._episode_count.
#
# Decay schedule:
#   SmartDrive: 4 decays over 1200 episodes = every 300 ep.
#   Our budget: ~400 episodes (200k steps), so every 75 episodes = 4 decays.
#   0.40 → 0.35 → 0.30 → 0.25 → 0.20 (floored at ACTION_STD_MIN=0.05)

import math
import torch
from stable_baselines3.common.callbacks import BaseCallback

from parameters import (
    ACTION_STD_INIT,
    ACTION_STD_MIN,
    ACTION_STD_DECAY,
    ACTION_STD_DECAY_FREQ,
)


class ExplorationDecayCallback(BaseCallback):
    """
    Decays policy action std every ACTION_STD_DECAY_FREQ episodes.
    Episode count is read from env info dict to survive checkpoint resumes.
    """

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self._current_std    = ACTION_STD_INIT
        # Track the last episode count seen, to detect new episodes
        self._last_ep_count  = 0
        # Track how many decay steps we've applied
        self._decays_applied = 0

    def _on_training_start(self):
        self._set_action_std(self._current_std)
        if self.verbose > 0:
            print(f"[ExplorationDecay] Initial sigma_noise = {self._current_std:.4f}")
            print(f"[ExplorationDecay] Decay: -{ACTION_STD_DECAY} every "
                  f"{ACTION_STD_DECAY_FREQ} episodes")

    def _on_step(self):
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [{}])

        for done, info in zip(dones, infos):
            if not done:
                continue

            # FIX-5: read episode count from env info (survives resume)
            ep_count = info.get("episode_count", self._last_ep_count)
            self._last_ep_count = ep_count

            # Compute how many decays should have been applied by now
            decays_due = ep_count // ACTION_STD_DECAY_FREQ
            while self._decays_applied < decays_due:
                new_std = max(
                    self._current_std - ACTION_STD_DECAY,
                    ACTION_STD_MIN,
                )
                self._current_std   = new_std
                self._decays_applied += 1
                self._set_action_std(new_std)

                if self.verbose > 0:
                    print(
                        f"\n[ExplorationDecay] Episode {ep_count}: "
                        f"sigma_noise -> {new_std:.4f}\n"
                    )

        return True

    def _set_action_std(self, std):
        """Set log_std on SB3 policy so it takes effect on next rollout."""
        policy = self.model.policy
        if not hasattr(policy, 'log_std'):
            return
        log_std_value = float(math.log(max(std, 1e-8)))
        with torch.no_grad():
            policy.log_std.requires_grad_(False)
            policy.log_std.fill_(log_std_value)
            policy.log_std.requires_grad_(True)