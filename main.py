# main.py
# Combined BLIP-FusePPO + SmartDrive training & evaluation entry point.
#
# PPO:         Stable Baselines3  (PyTorch)
# Policy:      MultiInputPolicy   (Dict obs — per-modality branches)
# Environment: CARLA              (SmartDrive structure)
# State:       BLIP + LiDAR + PID + Nav  (BLIP-FusePPO)
# Reward:      Hybrid nonlinear          (BLIP-FusePPO)
# Exploration: Variable sigma_noise decay (SmartDrive via callback)
# Logging:     TensorBoard + CSV         (per-episode training log)
#
# FIX: env_ref now actually stores the environment object so the
#      finally block can close it on interrupt or exception.

import os
import sys
import random
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from parameters import (
    TOWN, SEED,
    LEARNING_RATE, N_STEPS, BATCH_SIZE, N_EPOCHS,
    GAMMA, GAE_LAMBDA, CLIP_RANGE, ENT_COEF, VF_COEF, MAX_GRAD_NORM,
    TRAIN_TIMESTEPS, TEST_EPISODES, EPISODE_LENGTH,
    PPO_MODEL_PATH, LOG_PATH_TRAIN, LOG_PATH_TEST,
    CHECKPOINT_PATH, CHECKPOINT_LOAD, CHECKPOINT_SAVE_FREQ,
)
from environment          import CarlaEnv, ClientConnection
from state_builder        import policy_kwargs
from exploration_callback import ExplorationDecayCallback
from reward_callback      import RewardLoggingCallback
from csv_logging_callback import CSVLoggingCallback


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
    env = Monitor(env, LOG_PATH_TRAIN)

    os.makedirs(LOG_PATH_TRAIN, exist_ok=True)
    os.makedirs(PPO_MODEL_PATH, exist_ok=True)

    # ── SB3 PPO model ─────────────────────────────────────────────────
    loading_checkpoint = CHECKPOINT_LOAD

    if loading_checkpoint:
        import glob as _glob
        ckpt_files = sorted(
            _glob.glob(os.path.join(CHECKPOINT_PATH, 'rl_model_*_steps.zip')),
            key=lambda f: int(os.path.basename(f).split('_')[2])
        )
        load_path = ckpt_files[-1] if ckpt_files else (
            PPO_MODEL_PATH + '.zip'
            if os.path.exists(PPO_MODEL_PATH + '.zip') else None
        )
        if load_path is None:
            print("CHECKPOINT_LOAD=True but no checkpoint found — starting fresh.")
            loading_checkpoint = False
        else:
            print(f"Resuming from checkpoint: {load_path}")
            model = PPO.load(
                load_path,
                env=env,
                tensorboard_log=LOG_PATH_TRAIN,
            )
            print("Checkpoint loaded.")

    if not loading_checkpoint:
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
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    callbacks = CallbackList([
        ExplorationDecayCallback(verbose=1),
        RewardLoggingCallback(verbose=1),
        CSVLoggingCallback(log_dir=LOG_PATH_TRAIN, verbose=1),
        CheckpointCallback(
            save_freq          = CHECKPOINT_SAVE_FREQ,
            save_path          = CHECKPOINT_PATH,
            name_prefix        = "rl_model",
            save_replay_buffer = False,
            save_vecnormalize  = False,
            verbose            = 1,
        ),
    ])

    # ── Train ─────────────────────────────────────────────────────────
    print(f"\nTraining for {int(TRAIN_TIMESTEPS):,} timesteps ...\n")
    print(f"CSV log -> {LOG_PATH_TRAIN}/training_log.csv\n")
    model.learn(
        total_timesteps     = int(TRAIN_TIMESTEPS),
        callback            = callbacks,
        reset_num_timesteps = not loading_checkpoint,
        tb_log_name         = "BLIP_FusePPO",
        progress_bar        = True,
    )

    # ── Save ──────────────────────────────────────────────────────────
    model.save(PPO_MODEL_PATH)
    print(f"\nModel saved to {PPO_MODEL_PATH}")

    env.close()
    return env   # return so caller can hold the ref


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

    total_rewards    = []
    total_distances  = []
    total_deviations = []

    for episode in range(1, TEST_EPISODES + 1):
        obs       = env.reset()
        done      = False
        ep_reward = 0.0

        for _ in range(int(EPISODE_LENGTH)):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            if done:
                break

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
    print(
        f"RMSE Deviation: "
        f"{np.sqrt(np.mean(np.array(total_deviations)**2)):.4f}m"
    )

    env.close()
    return env


# ═══════════════════════════════════════════════════════════════════════ #
#  Entry point                                                           #
# ═══════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    # FIX: env_ref is now assigned the return value of train()/test()
    # so the finally block can close the environment on interrupt or crash.
    env_ref = None
    try:
        env_ref = train()
        # env_ref = test()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\nFatal error: {e}")
        raise
    finally:
        if env_ref is not None:
            try:
                env_ref.close()
            except Exception:
                pass
        print("Done.")