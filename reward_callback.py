# reward_callback.py
# SB3 callback for TensorBoard logging of custom metrics.
# Logs: episodic reward, distance covered, lane deviation,
#       average speed, current sigma noise.

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class RewardLoggingCallback(BaseCallback):
    """
    Logs custom driving metrics to TensorBoard every episode.
    Mirrors SmartDrive's summary_writer logging.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_rewards   = []
        self._episode_distances = []
        self._episode_deviations = []
        self._current_ep_reward = 0.0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [0.0])
        dones   = self.locals.get("dones",   [False])
        infos   = self.locals.get("infos",   [{}])

        for reward, done, info in zip(rewards, dones, infos):
            self._current_ep_reward += reward

            if done:
                ep = len(self._episode_rewards) + 1
                self._episode_rewards.append(self._current_ep_reward)

                dist = info.get("distance_covered", 0)
                dev  = info.get("center_lane_deviation", 0)
                self._episode_distances.append(dist)
                self._episode_deviations.append(dev)

                # Log to TensorBoard via SB3's logger
                self.logger.record("custom/episode_reward",    self._current_ep_reward)
                self.logger.record("custom/distance_covered",  dist)
                self.logger.record("custom/lane_deviation",    dev)
                self.logger.record("custom/mean_reward_10ep",
                                   float(np.mean(self._episode_rewards[-10:])))

                # Log sigma noise if available
                policy = self.model.policy
                if hasattr(policy, 'log_std'):
                    import torch, math
                    std = float(torch.exp(policy.log_std).mean().item())
                    self.logger.record("custom/sigma_noise", std)

                self.logger.dump(step=self.num_timesteps)

                if self.verbose > 0:
                    print(
                        f"Ep {ep:4d} | "
                        f"Reward: {self._current_ep_reward:7.2f} | "
                        f"Dist: {dist:5.0f}m | "
                        f"Dev: {dev:.4f}m"
                    )

                self._current_ep_reward = 0.0

        return True
