# parameters.py
# Combined BLIP-FusePPO + SmartDrive parameters
# PPO: Stable Baselines3 (PyTorch)
# Environment: CARLA (SmartDrive structure)
#
# FIX-A: LANE_RESET_DIST raised 85 → 120 px.
#        The Hough detector is noisy; 85 px was terminating valid episodes.
#        SmartDrive's equivalent metric threshold is 3.0 m (~120 px at this scale).
#
# FIX-B: LIDAR termination window doubled (20 → 40 steps, same count=10).
#        This changes the trigger from 50% → 25% of window, matching
#        SmartDrive's more forgiving early-training behaviour.
#
# FIX-C: ACTION_STD_DECAY_FREQ 300 → 75 episodes.
#        SmartDrive's 4 decay steps over 1200 episodes = every 300 ep.
#        Our budget is ~400 episodes (100k steps), so 4 decays = every 75 ep.
#
# FIX-D: W_BLIP disabled for first 200 episodes via BLIP_WARMUP_EPISODES.
#        The safe reference is a proxy image, not a real road frame.
#        Adding cosine-similarity noise in early training hurts convergence.
#
# FIX-E: checkpoint_frequency initial value 100 → 50.
#        Saves checkpoints more often so the agent recovers to a closer
#        restart point, exactly matching SmartDrive's checkpoint-reset strategy.

# ─────────────────────────────────────────────
# CAMERA / IMAGE
# ─────────────────────────────────────────────
IM_WIDTH  = 160
IM_HEIGHT = 80

# ─────────────────────────────────────────────
# BLIP ENCODER
# ─────────────────────────────────────────────
BLIP_MODEL_NAME      = "Salesforce/blip-image-captioning-large"
BLIP_EMBEDDING_DIM   = 768        # BERT-based text encoder output
BLIP_UPDATE_INTERVAL = 10         # regenerate every K steps, cache otherwise
BLIP_MAX_LENGTH      = 50
BLIP_CACHE_SIZE      = 100

# FIX-D: disable BLIP reward bonus for this many episodes at the start.
# Set to 0 to enable from the beginning (once you have a real reference frame).
BLIP_WARMUP_EPISODES = 200

# ─────────────────────────────────────────────
# OBSERVATION SPACE KEYS & DIMS
# ─────────────────────────────────────────────
OBS_KEY_IMAGE  = "image"
OBS_KEY_BLIP   = "blip"
OBS_KEY_LIDAR  = "lidar"
OBS_KEY_PID    = "pid_correction"
OBS_KEY_NAV    = "navigation"

LIDAR_DIM = 180
PID_DIM   = 1
NAV_DIM   = 5

# ─────────────────────────────────────────────
# ACTION SPACE
# ─────────────────────────────────────────────
ACTION_DIM = 2   # [steering, throttle]

# ─────────────────────────────────────────────
# SB3 PPO HYPERPARAMETERS
# ─────────────────────────────────────────────
LEARNING_RATE = 3e-4
N_STEPS       = 2048
BATCH_SIZE    = 64
N_EPOCHS      = 10
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
CLIP_RANGE    = 0.2
ENT_COEF      = 0.0
VF_COEF       = 0.5
MAX_GRAD_NORM = 0.5
SEED          = 42

# FIX-C: decay every 75 episodes so 4 full decay steps fit in ~400-episode run
ACTION_STD_INIT       = 0.4
ACTION_STD_MIN        = 0.05
ACTION_STD_DECAY      = 0.05
ACTION_STD_DECAY_FREQ = 75        # was 300 — now 4 decays over 300-episode run

# ─────────────────────────────────────────────
# REWARD WEIGHTS
# ─────────────────────────────────────────────
W_LANE   = 0.3
W_LIDAR  = 0.3
W_SPEED  = 0.2
W_CENTER = 0.2
W_BLIP   = 0.2   # active only after BLIP_WARMUP_EPISODES (FIX-D)

REWARD_TERMINAL = -10.0
REWARD_FAIL     = -3.0

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
LANE_NORM       = 100.0   # pixels
CENTER_MAX_DIST = 80.0    # pixels
CENTER_K        = 2.5

# FIX-A: raised from 85 → 120 px.
# Hough detector noise was causing valid episodes to terminate.
# 120 px ≈ 3.0 m, matching SmartDrive's max_distance_from_center.
LANE_RESET_DIST = 120.0   # pixels — was 85

# Lane centre — real-world scale used by reward.py
MAX_DISTANCE_FROM_CENTER = 3.0   # metres

# Speed
TARGET_SPEED = 20.0   # km/h
MAX_SPEED    = 35.0
MIN_SPEED    = 15.0

REWARD_CLIP = 1.0

# FIX-B: window doubled (20→40), count unchanged (10).
# Trigger threshold: 10/40 = 25% instead of 10/20 = 50%.
# Prevents over-eager termination when agent brushes obstacles in early training.
LIDAR_BELOW_THRESH_COUNT = 10
LIDAR_WINDOW_STEPS       = 40    # was 20

# ─────────────────────────────────────────────
# TRAINING PARAMETERS
# ─────────────────────────────────────────────
TRAIN_TIMESTEPS      = 100_000
EPISODE_LENGTH       = 75000
TEST_EPISODES        = 30
NO_OF_TEST_EPISODES  = 10
CHECKPOINT_SAVE_FREQ = 10_000

# ─────────────────────────────────────────────
# SIMULATION / CARLA
# ─────────────────────────────────────────────
TOWN                 = 'Town02'
CAR_NAME             = 'model3'
NUMBER_OF_VEHICLES   = 30
NUMBER_OF_PEDESTRIAN = 10
CONTINUOUS_ACTION    = True
VISUAL_DISPLAY       = False

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
CHECKPOINT_LOAD = False