# main.py
# Combined BLIP-FusePPO + SmartDrive training & evaluation entry point.
#
# PPO:         Stable Baselines3  (PyTorch)
# Policy:      MultiInputPolicy   (Dict obs — per-modality branches)
# Environment: CARLA              (SmartDrive structure)
# State:       BLIP + LiDAR + PID + Nav  (BLIP-FusePPO)
# Reward:      Hybrid nonlinear          (BLIP-FusePPO)
# Exploration: Variable σ_noise decay   (SmartDrive via callback)
# Augmentation:Symmetric flip           (BLIP-FusePPO, in env.step)

import os
import sys
import random
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor

from parameters import (
    TOWN, SEED,
    LEARNING_RATE, N_STEPS, BATCH_SIZE, N_EPOCHS,
    GAMMA, GAE_LAMBDA, CLIP_RANGE, ENT_COEF, VF_COEF, MAX_GRAD_NORM,
    TRAIN_TIMESTEPS, TEST_EPISODES, EPISODE_LENGTH,
    PPO_MODEL_PATH, LOG_PATH_TRAIN, LOG_PATH_TEST,
    CHECKPOINT_LOAD,
)
from environment         import CarlaEnv, ClientConnection
from state_builder       import policy_kwargs
from exploration_callback import ExplorationDecayCallback
from reward_callback     import RewardLoggingCallback


# ═══════════════════════════════════════════════════════════════════════ #
#  Seeding                                                               #
# ═══════════════════════════════════════════════════════════════════════ #

def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════════ #
#  Training                                                              #
# ═══════════════════════════════════════════════════════════════════════ #

def train():
    set_seeds()

    # ── CARLA connection ─────────────────────────────────────────────
    try:
        client, world = ClientConnection().setup()
    except ConnectionError as e:
        print(f"CARLA refused: {e}")
        sys.exit(1)

    # ── Gym environment ───────────────────────────────────────────────
    env = CarlaEnv(client, world)
    env = Monitor(env, LOG_PATH_TRAIN)   # wraps env for SB3 episode stats

    os.makedirs(LOG_PATH_TRAIN,  exist_ok=True)
    os.makedirs(PPO_MODEL_PATH,  exist_ok=True)

    # ── SB3 PPO model ─────────────────────────────────────────────────
    if CHECKPOINT_LOAD and os.path.exists(PPO_MODEL_PATH + '.zip'):
        print(f"Loading model from {PPO_MODEL_PATH} ...")
        model = PPO.load(
            PPO_MODEL_PATH,
            env=env,
            tensorboard_log=LOG_PATH_TRAIN,
        )
        print("Model loaded.")
    else:
        print("Creating new PPO model ...")
        model = PPO(
            policy            = "MultiInputPolicy",
            env               = env,
            policy_kwargs     = policy_kwargs,
            learning_rate     = LEARNING_RATE,
            n_steps           = N_STEPS,
            batch_size        = BATCH_SIZE,
            n_epochs          = N_EPOCHS,
            gamma             = GAMMA,
            gae_lambda        = GAE_LAMBDA,
            clip_range        = CLIP_RANGE,
            ent_coef          = ENT_COEF,
            vf_coef           = VF_COEF,
            max_grad_norm     = MAX_GRAD_NORM,
            use_sde           = False,
            tensorboard_log   = LOG_PATH_TRAIN,
            verbose           = 1,
            seed              = SEED,
            device            = "auto",
        )

    # ── Callbacks ─────────────────────────────────────────────────────
    callbacks = CallbackList([
        ExplorationDecayCallback(verbose=1),   # SmartDrive σ_noise decay
        RewardLoggingCallback(verbose=1),      # custom TensorBoard metrics
    ])

    # ── Train ─────────────────────────────────────────────────────────
    print(f"\nTraining for {int(TRAIN_TIMESTEPS):,} timesteps ...\n")
    model.learn(
        total_timesteps   = int(TRAIN_TIMESTEPS),
        callback          = callbacks,
        reset_num_timesteps = not CHECKPOINT_LOAD,
        tb_log_name       = "BLIP_FusePPO",
        progress_bar      = True,
    )

    # ── Save ──────────────────────────────────────────────────────────
    model.save(PPO_MODEL_PATH)
    print(f"\nModel saved to {PPO_MODEL_PATH}")

    env.close()


# ═══════════════════════════════════════════════════════════════════════ #
#  Testing                                                               #
# ═══════════════════════════════════════════════════════════════════════ #

def test():
    set_seeds()

    try:
        client, world = ClientConnection().setup()
    except ConnectionError as e:
        print(f"CARLA refused: {e}")
        sys.exit(1)

    env = CarlaEnv(client, world)
    os.makedirs(LOG_PATH_TEST, exist_ok=True)

    print(f"Loading model from {PPO_MODEL_PATH} ...")
    model = PPO.load(PPO_MODEL_PATH, env=env)
    print("Model loaded.\n")

    episode           = 0
    total_rewards     = []
    total_distances   = []
    total_deviations  = []

    while episode < TEST_EPISODES:
        obs   = env.reset()
        done  = False
        ep_reward = 0.0

        for _ in range(int(EPISODE_LENGTH)):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            if done:
                break

        episode += 1
        dist = info.get("distance_covered", 0)
        dev  = info.get("center_lane_deviation", 0)
        total_rewards.append(ep_reward)
        total_distances.append(dist)
        total_deviations.append(dev)

        print(
            f"Ep {episode:3d} | "
            f"Reward: {ep_reward:7.2f} | "
            f"Dist: {dist:5.0f}m | "
            f"Dev: {dev:.4f}m"
        )

    print("\n── Test Summary ──────────────────────────────")
    print(f"Mean Reward:    {np.mean(total_rewards):.2f}")
    print(f"Mean Distance:  {np.mean(total_distances):.0f}m")
    print(f"Mean Deviation: {np.mean(total_deviations):.4f}m")
    print(f"RMSE Deviation: {np.sqrt(np.mean(np.array(total_deviations)**2)):.4f}m")

    env.close()


# ═══════════════════════════════════════════════════════════════════════ #
#  Entry point                                                           #
# ═══════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    try:
        train()
        # test()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    finally:
        print("Done.")
