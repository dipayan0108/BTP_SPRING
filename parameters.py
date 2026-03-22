# parameters.py
# OPTIMISED for 2000+ episode training run — best results configuration.
#
# Key changes from baseline:
#   TRAIN_TIMESTEPS     : 500,000  (guarantees 2000+ eps at any episode length)
#   BLIP_UPDATE_INTERVAL: 20       (was 10 — halves BLIP overhead, ~30% speedup)
#   BLIP_CACHE_SIZE     : 200      (larger cache for longer runs)
#   BATCH_SIZE          : 128      (was 64 — more stable gradients)
#   N_EPOCHS            : 15       (was 10 — more gradient steps per rollout)
#   ENT_COEF            : 0.005    (was 0.0 — entropy bonus prevents premature convergence)
#   LEARNING_RATE       : 2.5e-4   (was 3e-4 — lower = more stable long run)
#   ACTION_STD_DECAY_FREQ: 250     (was 300 — sigma hits min ~ep 1750)

# ─────────────────────────────────────────────
# CAMERA / IMAGE
# ─────────────────────────────────────────────
IM_WIDTH  = 160
IM_HEIGHT = 80

# ─────────────────────────────────────────────
# BLIP ENCODER
# ─────────────────────────────────────────────
BLIP_MODEL_NAME      = "Salesforce/blip-image-captioning-large"
BLIP_EMBEDDING_DIM   = 768
BLIP_UPDATE_INTERVAL = 20         # was 10 — halves BLIP calls, ~30% faster training
BLIP_MAX_LENGTH      = 50
BLIP_CACHE_SIZE      = 200        # was 100 — larger cache for longer runs

# ─────────────────────────────────────────────
# OBSERVATION SPACE KEYS & DIMS
# ─────────────────────────────────────────────
OBS_KEY_IMAGE  = "image"
OBS_KEY_LIDAR  = "lidar"
OBS_KEY_PID    = "pid_correction"
OBS_KEY_NAV    = "navigation"
OBS_KEY_BLIP   = "blip"

LIDAR_DIM = 180
PID_DIM   = 1
NAV_DIM   = 5

# ─────────────────────────────────────────────
# ACTION SPACE
# ─────────────────────────────────────────────
ACTION_DIM = 2   # [steering, throttle]

# ─────────────────────────────────────────────
# SB3 PPO HYPERPARAMETERS — optimised for long run
# ─────────────────────────────────────────────
LEARNING_RATE = 2.5e-4    # was 3e-4 — lower LR = more stable over long training
N_STEPS       = 2048      # rollout buffer — covers ~10 episodes
BATCH_SIZE    = 128       # was 64 — larger minibatch = more stable gradients
N_EPOCHS      = 15        # was 10 — more gradient passes per rollout
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
CLIP_RANGE    = 0.2
ENT_COEF      = 0.005     # was 0.0 — prevents premature convergence early on
VF_COEF       = 0.5
MAX_GRAD_NORM = 0.5
SEED          = 42

# ─────────────────────────────────────────────
# EXPLORATION NOISE
# Decay schedule: sigma decays every 250 eps
#   ep 250  -> 0.35
#   ep 500  -> 0.30
#   ep 750  -> 0.25
#   ep 1000 -> 0.20
#   ep 1250 -> 0.15
#   ep 1500 -> 0.10
#   ep 1750 -> 0.05 (minimum — fine-grained control for final 250 eps)
# ─────────────────────────────────────────────
ACTION_STD_INIT       = 0.4
ACTION_STD_MIN        = 0.05
ACTION_STD_DECAY      = 0.05
ACTION_STD_DECAY_FREQ = 250    # was 300

# ─────────────────────────────────────────────
# REWARD WEIGHTS
# ─────────────────────────────────────────────
W_LANE   = 0.3
W_LIDAR  = 0.3
W_SPEED  = 0.2
W_CENTER = 0.2

# LiDAR reward thresholds (metres)
LIDAR_DMID         = 8.0
LIDAR_DLOW         = 4.0
LIDAR_DCRIT        = 2.8
LIDAR_DFAIL        = 2.0
LIDAR_B1           = 5.0
LIDAR_B2           = 10.0
LIDAR_B3           = 2.0
LIDAR_BONUS        = 5.0
LIDAR_BONUS_RANGES = [(3.0, 4.0), (8.0, 10.0)]

# Lane / center thresholds
LANE_NORM       = 100.0
CENTER_MAX_DIST = 80.0
CENTER_K        = 2.5
LANE_RESET_DIST = 85.0

# Speed
TARGET_SPEED = 20.0   # km/h
MAX_SPEED    = 35.0
MIN_SPEED    = 15.0

# Reward clipping
REWARD_CLIP = 1.0
REWARD_FAIL = -3.0

# LiDAR termination counters
LIDAR_BELOW_THRESH_COUNT = 10
LIDAR_WINDOW_STEPS       = 20

# ─────────────────────────────────────────────
# TRAINING PARAMETERS
# ─────────────────────────────────────────────
# 500,000 steps guarantees 2000+ episodes:
#   avg 100 steps/ep -> ~5,000 episodes
#   avg 250 steps/ep -> ~2,000 episodes
# Estimated wall time at 11 it/s with BLIP_UPDATE_INTERVAL=20: ~9-10 hours
TRAIN_TIMESTEPS     = 500_000
EPISODE_LENGTH      = 75000
TEST_EPISODES       = 30
NO_OF_TEST_EPISODES = 10

# ─────────────────────────────────────────────
# SIMULATION / CARLA
# ─────────────────────────────────────────────
TOWN                 = 'Town02'
CAR_NAME             = 'model3'
NUMBER_OF_VEHICLES   = 30
NUMBER_OF_PEDESTRIAN = 10
CONTINUOUS_ACTION    = True
VISUAL_DISPLAY       = True   # keep False during training — saves ~2 it/s

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
RESULTS_PATH    = 'Results_BLIP'
PPO_MODEL_PATH  = f'{RESULTS_PATH}/ppo_model'
CHECKPOINT_PATH = f'{RESULTS_PATH}/checkpoints'
LOG_PATH_TRAIN  = f'{RESULTS_PATH}/runs/train'
LOG_PATH_TEST   = f'{RESULTS_PATH}/runs/test'
TEST_IMAGES     = f'{RESULTS_PATH}/test_images'

# ─────────────────────────────────────────────
# MISC
# ─────────────────────────────────────────────
CHECKPOINT_LOAD = True  # set True to resume from latest checkpoint

# Save a checkpoint every N environment steps.
# At 11 it/s: 25,000 steps ≈ every 38 minutes
# At 15 it/s: 25,000 steps ≈ every 28 minutes
# Keeps last ~20 checkpoints (500k / 25k = 20 files × ~100MB each)
CHECKPOINT_SAVE_FREQ = 25_000