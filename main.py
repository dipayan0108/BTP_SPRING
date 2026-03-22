# main.py
# Combined BLIP-FusePPO + SmartDrive training & evaluation entry point.
#
# PPO:         Stable Baselines3  (PyTorch)
# Policy:      MultiInputPolicy   (Dict obs — per-modality branches)
# Environment: CARLA              (SmartDrive structure)
# State:       BLIP + LiDAR + PID + Nav  (BLIP-FusePPO)
# Reward:      Hybrid nonlinear          (BLIP-FusePPO)
# Exploration: Variable σ_noise decay   (SmartDrive via callback)
# Logging:     TensorBoard + CSV        (per-episode training log)

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
    env = Monitor(env, LOG_PATH_TRAIN)   # wraps env for SB3 episode stats

    os.makedirs(LOG_PATH_TRAIN,  exist_ok=True)
    os.makedirs(PPO_MODEL_PATH,  exist_ok=True)

    # ── SB3 PPO model ─────────────────────────────────────────────────
    if CHECKPOINT_LOAD:
        # Auto-find the latest checkpoint in CHECKPOINT_PATH
        # Files are named: rl_model_<steps>_steps.zip
        import glob as _glob
        ckpt_files = sorted(
            _glob.glob(os.path.join(CHECKPOINT_PATH, 'rl_model_*_steps.zip')),
            key=lambda f: int(os.path.basename(f).split('_')[2])
        )
        load_path = ckpt_files[-1] if ckpt_files else (
            PPO_MODEL_PATH + '.zip' if os.path.exists(PPO_MODEL_PATH + '.zip')
            else None
        )
        if load_path is None:
            print("CHECKPOINT_LOAD=True but no checkpoint or model found — starting fresh.")
            load_path = None

        if load_path:
            print(f"Resuming from checkpoint: {load_path}")
            model = PPO.load(
                load_path,
                env=env,
                tensorboard_log=LOG_PATH_TRAIN,
            )
            print("Checkpoint loaded.")
        else:
            CHECKPOINT_LOAD = False   # fall through to fresh model below
    if not CHECKPOINT_LOAD:
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
        ExplorationDecayCallback(verbose=1),        # SmartDrive σ_noise decay
        RewardLoggingCallback(verbose=1),           # TensorBoard custom metrics
        CSVLoggingCallback(                         # per-episode CSV log
            log_dir=LOG_PATH_TRAIN,
            verbose=1,
        ),
        # Saves model every CHECKPOINT_SAVE_FREQ steps into CHECKPOINT_PATH/
        # e.g. rl_model_50000_steps.zip, rl_model_100000_steps.zip ...
        # On crash or interrupt, resume from the latest checkpoint by
        # setting CHECKPOINT_LOAD = True in parameters.py and pointing
        # PPO_MODEL_PATH to the desired checkpoint file.
        CheckpointCallback(
            save_freq=CHECKPOINT_SAVE_FREQ,
            save_path=CHECKPOINT_PATH,
            name_prefix="rl_model",
            save_replay_buffer=False,
            save_vecnormalize=False,
            verbose=1,
        ),
    ])

    # ── Train ─────────────────────────────────────────────────────────
    print(f"\nTraining for {int(TRAIN_TIMESTEPS):,} timesteps ...\n")
    print(f"CSV log → {LOG_PATH_TRAIN}/training_log.csv\n")
    model.learn(
        total_timesteps     = int(TRAIN_TIMESTEPS),
        callback            = callbacks,
        reset_num_timesteps = not CHECKPOINT_LOAD,
        tb_log_name         = "BLIP_FusePPO",
        progress_bar        = True,
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
    env_ref = None   # hold ref so finally block can close it on interrupt
    try:
        train()
        # test()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)
    finally:
        print("Done.")