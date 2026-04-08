# parameters.py
# SmartDrive + BLIP-FusePPO hybrid — CARLA + SB3 PPO
#
# FINAL VERSION — aligned to SmartDrive paper exactly
#
# KEY FIXES vs all previous versions:
#
#   FIX-A: MAX_DISTANCE_FROM_CENTER 3.5 → 2.0m
#       Car was surviving at 3.0m edge → r_base=0.14/step local optimum.
#       At 2.0m: must stay within ±1m centre or reward collapses.
#
#   FIX-B: TRAIN_TIMESTEPS 200k → 1_000_000
#       Previous run: ~72k effective steps (3% of SmartDrive's 2.3M).
#
#   FIX-C: ACTION_STD_DECAY_FREQ 75 → 300  ← CRITICAL
#       SmartDrive decays every 300 episodes, NOT every 75.
#       Going to σ=0.05 caused the regression at ep525.
#
#   FIX-D: ACTION_STD_MIN 0.05 → 0.20  ← CRITICAL
#       SmartDrive NEVER goes below σ=0.20.
#       σ=0.05 over-exploited a bad local optimum for 460 episodes.
#       Schedule: 0.40→0.35→0.30→0.25→0.20 (stays at 0.20)
#
#   FIX-E: W_SPEED 0.2 → 0.4
#       At W_SPEED=0.2, speed penalty at 3.5kmh = only -0.14/step.
#       Car was happy to crawl. At W_SPEED=0.4, crawling costs -0.28/step.
#       Forces agent to learn to throttle to avoid speed penalty.
#
#   FIX-F: TARGET_SPEED 20 → 20 (REVERTED)
#       SmartDrive used 20 km/h. Raising to 25 changed the reward scale
#       and made it harder to compare results. Keep at 20.
#
#   FIX-G: CHECKPOINT_LOAD = False
#       No valid checkpoint from previous run (wrong sigma schedule).
#       Training from scratch with correct hyperparameters.

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
BLIP_UPDATE_INTERVAL = 10
BLIP_MAX_LENGTH      = 50
BLIP_CACHE_SIZE      = 100
BLIP_WARMUP_EPISODES = 50

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
N_STEPS       = 512
BATCH_SIZE    = 64
N_EPOCHS      = 10
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
CLIP_RANGE    = 0.2
ENT_COEF      = 0.01
VF_COEF       = 0.5
MAX_GRAD_NORM = 0.5
SEED          = 42

# ─────────────────────────────────────────────
# EXPLORATION NOISE  — match SmartDrive exactly
# ─────────────────────────────────────────────
ACTION_STD_INIT  = 0.4

# FIX-D: minimum 0.20 — SmartDrive never goes below this
ACTION_STD_MIN   = 0.20

ACTION_STD_DECAY = 0.05

# FIX-C: decay every 300 episodes — exactly SmartDrive's schedule
# Decay path: ep300→0.35, ep600→0.30, ep900→0.25, ep1200→0.20 (stops)
# Total: 4 decays over 1200 episodes, then stays at 0.20 forever
ACTION_STD_DECAY_FREQ = 300

# ─────────────────────────────────────────────
# REWARD WEIGHTS
# ─────────────────────────────────────────────
W_LANE   = 0.3
W_LIDAR  = 0.3

# FIX-E: 0.2 → 0.4 — crawling at 3.5kmh must cost more than survival gain
W_SPEED  = 0.4

W_CENTER = 0.2

REWARD_TERMINAL = -10.0   # SmartDrive value
REWARD_FAIL     = -3.0    # soft fail (env exception)

# ─────────────────────────────────────────────
# REWARD: LIDAR TERMS
# ─────────────────────────────────────────────
LIDAR_DCRIT        = 2.0
LIDAR_DLOW         = 3.0
LIDAR_SAFE1_LO     = 3.0
LIDAR_SAFE1_HI     = 5.0
LIDAR_SAFE2_LO     = 7.0
LIDAR_SAFE2_HI     = 10.0
LIDAR_DFAIL        = 2.0
LIDAR_BONUS        = 1.0
LIDAR_BONUS2       = 0.6

# ─────────────────────────────────────────────
# REWARD: LANE / CENTER TERMS
# ─────────────────────────────────────────────
LANE_NORM       = 100.0
CENTER_MAX_DIST = 80.0
CENTER_K        = 2.5
REWARD_CLIP     = 2.0
LANE_RESET_DIST = 120.0

# FIX-A: 3.5 → 2.0m — forces genuine lane-keeping, breaks edge-hugging
# centering_factor at 1.0m off: (2.0-1.0)/2.0 = 0.50 (was 0.71)
# centering_factor at 1.5m off: (2.0-1.5)/2.0 = 0.25 (was 0.57)
# centering_factor at 2.0m off: 0.00 (episode ends)
MAX_DISTANCE_FROM_CENTER = 2.0   # metres

# ─────────────────────────────────────────────
# SPEED — match SmartDrive
# ─────────────────────────────────────────────
TARGET_SPEED = 20.0   # km/h — SmartDrive's value (17.73 avg achieved)
MAX_SPEED    = 35.0   # km/h
MIN_SPEED    = 5.0    # km/h

# ─────────────────────────────────────────────
# LIDAR TERMINATION WINDOW
# ─────────────────────────────────────────────
LIDAR_BELOW_THRESH_COUNT = 10
LIDAR_WINDOW_STEPS       = 40

# ─────────────────────────────────────────────
# CARLA / ENVIRONMENT
# ─────────────────────────────────────────────
TOWN                 = "Town02"
CAR_NAME             = "vehicle.lincoln.mkz_2017"
NUMBER_OF_PEDESTRIAN = 0
VISUAL_DISPLAY       = False
EPISODE_LENGTH       = 10000

# ─────────────────────────────────────────────
# TRAINING / PATHS
# ─────────────────────────────────────────────
# FIX-B: 200k → 1M
# At avg 150 steps/ep (target with correct training): ~6600 episodes
# At avg 500 steps/ep (when agent learns 100m+):      ~2000 episodes
# Either way: far more than SmartDrive's 1200 episodes
TRAIN_TIMESTEPS    = 1_000_000

TEST_EPISODES      = 10    # match SmartDrive's Table IV (10 episodes)

# FIX-G: False — previous checkpoint used wrong sigma schedule
CHECKPOINT_LOAD    = False

CHECKPOINT_SAVE_FREQ = 10_000

PPO_MODEL_PATH  = "Results_BLIP/model/blip_fuseppo"
LOG_PATH_TRAIN  = "Results_BLIP/logs/train"
LOG_PATH_TEST   = "Results_BLIP/logs/test"
CHECKPOINT_PATH = "Results_BLIP/checkpoints"
RESULTS_PATH    = "Results_BLIP"