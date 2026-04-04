# exploration_callback.py
# Implements SmartDrive's variable exploration noise decay
# as an SB3 BaseCallback.
#
# FIX (original): log_std was frozen (requires_grad=False) permanently
#   after each manual decay. This prevented PPO's optimizer from ever
#   learning the std between decay steps, effectively disabling all
#   log_std gradient updates for the entire training run.
#
# FIX (this version): log_std.requires_grad_(False) is used only as a
#   momentary context to safely write the value. It is immediately
#   re-enabled with requires_grad_(True) so the optimizer can continue
#   learning the std between scheduled decay steps.

import math
import numpy as np
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
    Decays the policy's action std (sigma_noise) every
    ACTION_STD_DECAY_FREQ episodes, from ACTION_STD_INIT down to
    ACTION_STD_MIN.

    Replicates SmartDrive's variable exploration noise strategy within
    SB3's callback framework.

    The decayed std is written to policy.log_std in-place without
    permanently freezing the parameter, so PPO's optimizer can still
    learn the std between scheduled decay events.
    """

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self._current_std   = ACTION_STD_INIT
        self._episode_count = 0

    def _on_training_start(self):
        self._set_action_std(self._current_std)
        if self.verbose > 0:
            print(
                f"[ExplorationDecay] Initial sigma_noise = {self._current_std}"
            )

    def _on_step(self):
        dones = self.locals.get("dones", [])
        for done in dones:
            if done:
                self._episode_count += 1

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
        """
        Writes log(std) into policy.log_std in-place.

        FIX: requires_grad is disabled only during the write, then
        immediately re-enabled so PPO's optimizer can still update
        log_std between scheduled decay steps.
        """
        policy = self.model.policy
        if not hasattr(policy, 'log_std'):
            if self.verbose > 0:
                print(
                    "[ExplorationDecay] Warning: "
                    "policy has no log_std attribute — skipping."
                )
            return

        log_std_value = float(math.log(max(std, 1e-8)))
        with torch.no_grad():
            # Temporarily disable grad tracking to allow in-place fill
            policy.log_std.requires_grad_(False)
            policy.log_std.fill_(log_std_value)
            # FIX: re-enable so the PPO optimizer can continue learning
            # the std between scheduled decay steps
            policy.log_std.requires_grad_(True)