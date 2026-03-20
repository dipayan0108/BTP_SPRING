# exploration_callback.py
# Implements SmartDrive's variable exploration noise decay
# as an SB3 BaseCallback.
#
# In SB3, action std is controlled via log_std parameters inside
# the policy network. We decay them manually every N episodes
# exactly as SmartDrive does.

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

    This replicates SmartDrive's variable exploration noise strategy
    within SB3's callback framework.
    """

    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self._current_std    = ACTION_STD_INIT
        self._episode_count  = 0

    def _on_step(self) -> bool:
        # Check if any episode just ended
        dones = self.locals.get("dones", [])
        for done in dones:
            if done:
                self._episode_count += 1

                # Decay every N episodes
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
        Directly sets the log_std parameter of the SB3 actor network.
        SB3's ActorCriticPolicy stores log_std as a nn.Parameter.
        """
        policy = self.model.policy
        with torch.no_grad():
            if hasattr(policy, 'log_std'):
                new_log_std = torch.full_like(
                    policy.log_std,
                    fill_value=float(np.log(std))
                )
                policy.log_std.copy_(new_log_std)
            else:
                print("[ExplorationDecay] Warning: "
                      "policy has no log_std attribute.")

    def _on_training_start(self) -> None:
        # Initialise std at training start
        self._set_action_std(self._current_std)
        if self.verbose > 0:
            print(f"[ExplorationDecay] Initial σ_noise = {self._current_std}")
