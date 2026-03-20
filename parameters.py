# parameters.py
# Combined BLIP-FusePPO + SmartDrive parameters
# PPO: Stable Baselines3 (PyTorch)
# Environment: CARLA (SmartDrive structure)

# ─────────────────────────────────────────────
# CAMERA / IMAGE
# ─────────────────────────────────────────────
IM_WIDTH  = 160
IM_HEIGHT = 80

# ─────────────────────────────────────────────
# BLIP ENCODER
# ─────────────────────────────────────────────
BLIP_MODEL_NAME      = "Salesforce/blip-image-captioning-large"
BLIP_EMBEDDING_DIM   = 768        # full BERT-based text encoder output
BLIP_UPDATE_INTERVAL = 10         # regenerate every K steps, cache otherwise
BLIP_MAX_LENGTH      = 50
BLIP_CACHE_SIZE      = 100

# ─────────────────────────────────────────────
# OBSERVATION SPACE KEYS & DIMS
# Dict space — one key per modality
# ─────────────────────────────────────────────
OBS_KEY_IMAGE  = "image"           # (IM_HEIGHT, IM_WIDTH, 3) float32 [0,1]
OBS_KEY_LIDAR  = "lidar"           # (180,)                   float32 [0,1]
OBS_KEY_PID    = "pid_correction"  # (1,)                     float32
OBS_KEY_NAV    = "navigation"      # (5,)                     float32

LIDAR_DIM = 180
PID_DIM   = 1
NAV_DIM   = 5    # [throttle, velocity, norm_vel, norm_dist, norm_angle]

# ─────────────────────────────────────────────
# ACTION SPACE
# ─────────────────────────────────────────────
ACTION_DIM = 2   # [steering, throttle]

# ─────────────────────────────────────────────
# SB3 PPO HYPERPARAMETERS  (BLIP-FusePPO values)
# ─────────────────────────────────────────────
LEARNING_RATE = 3e-4
N_STEPS       = 2048    # rollout steps before each PPO update
BATCH_SIZE    = 64
N_EPOCHS      = 10      # gradient epochs per PPO update
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
CLIP_RANGE    = 0.2
ENT_COEF      = 0.0
VF_COEF       = 0.5
MAX_GRAD_NORM = 0.5
SEED          = 42

# Variable exploration noise  (SmartDrive strategy via SB3 callback)
ACTION_STD_INIT       = 0.4
ACTION_STD_MIN        = 0.05
ACTION_STD_DECAY      = 0.05
ACTION_STD_DECAY_FREQ = 300    # decay every N episodes

# ─────────────────────────────────────────────
# REWARD WEIGHTS  (BLIP-FusePPO hybrid reward)
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
LANE_NORM       = 100.0   # pixels
CENTER_MAX_DIST = 80.0    # pixels
CENTER_K        = 2.5
LANE_RESET_DIST = 85.0    # pixels → episode termination

# Speed
TARGET_SPEED = 20.0   # km/h
MAX_SPEED    = 35.0
MIN_SPEED    = 15.0

# Reward clipping
REWARD_CLIP = 1.0
REWARD_FAIL = -3.0

# LiDAR episode termination counters
LIDAR_BELOW_THRESH_COUNT = 10
LIDAR_WINDOW_STEPS       = 20

# ─────────────────────────────────────────────
# TRAINING PARAMETERS
# ─────────────────────────────────────────────
TRAIN_TIMESTEPS     = 100_000
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
VISUAL_DISPLAY       = False   # True enables pygame third-person view

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
