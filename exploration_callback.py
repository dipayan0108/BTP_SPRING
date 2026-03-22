# exploration_callback.py
# Implements SmartDrive's variable exploration noise decay
# as an SB3 BaseCallback.
#
# FIX: log_std is frozen (requires_grad=False) after each manual decay.
# Without this, SB3's optimizer overwrites the manually set value every
# PPO update, making the decay have no lasting effect.

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
    Decays the policy's action std (σ_noise) every ACTION_STD_DECAY_FREQ
    episodes, from ACTION_STD_INIT down to ACTION_STD_MIN.

    Replicates SmartDrive's variable exploration noise strategy within
    SB3's callback framework.

    FIX: After setting log_std, requires_grad is set to False so that
    PPO gradient updates do not overwrite the manually decayed value.
    """

    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self._current_std   = ACTION_STD_INIT
        self._episode_count = 0

    def _on_step(self) -> bool:
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
                            f"σ_noise → {self._current_std:.4f}\n"
                        )
        return True

    def _set_action_std(self, std: float):
        """
        Sets log_std on the SB3 actor and freezes it from gradient updates.

        FIX: policy.log_std.requires_grad_(False) prevents the PPO optimizer
        from overwriting the decayed value during the next policy update.
        """
        policy = self.model.policy
        with torch.no_grad():
            if hasattr(policy, 'log_std'):
                policy.log_std.fill_(float(np.log(std)))
                # FIX: freeze so the optimizer cannot undo the decay
                policy.log_std.requires_grad_(False)
            else:
                print("[ExplorationDecay] Warning: "
                      "policy has no log_std attribute.")

    def _on_training_start(self) -> None:
        self._set_action_std(self._current_std)
        if self.verbose > 0:
            print(f"[ExplorationDecay] Initial σ_noise = {self._current_std}")