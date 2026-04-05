# parameters.py
# SmartDrive + BLIP-FusePPO hybrid — CARLA + SB3 PPO
#
# CHANGES vs previous version:
#   FIX-1: W_BLIP removed from reward. BLIP is state-only per paper design.
#           The paper's core claim: inject BLIP into state, not reward.
#   FIX-2: N_STEPS 2048 → 512. More frequent policy updates (~2 eps/update
#           vs ~8 eps/update). Matches SmartDrive's per-episode update cadence.
#   FIX-3: ENT_COEF 0.0 → 0.01. Non-zero entropy keeps exploration alive
#           alongside sigma decay callback. Critical for policy not collapsing.
#   FIX-4: BLIP_WARMUP_EPISODES 200 → 50. With ~400-episode budget, 200ep
#           warmup wastes half of training with uninformative BLIP embeddings.
#   FIX-5: ExplorationDecay callback episode counter now synced from env.
#           (handled in exploration_callback.py)
#   FIX-6: Augmented reward recomputed (handled in environment.py).
#   FIX-7: LiDAR reward term wired into compute_reward() (reward.py).

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
BLIP_UPDATE_INTERVAL = 10        # regenerate every K steps, cache otherwise
BLIP_MAX_LENGTH      = 50
BLIP_CACHE_SIZE      = 100

# FIX-4: reduced from 200 → 50.
# With ~400 total episodes, 50ep warmup lets the agent learn with a
# meaningful BLIP state for 350 episodes instead of only 200.
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
# FIX-2: 2048 → 512. Policy updates every ~2 episodes instead of ~8.
N_STEPS       = 512
BATCH_SIZE    = 64
N_EPOCHS      = 10
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
CLIP_RANGE    = 0.2
# FIX-3: 0.0 → 0.01. Prevents policy collapsing to deterministic too early.
ENT_COEF      = 0.01
VF_COEF       = 0.5
MAX_GRAD_NORM = 0.5
SEED          = 42

# Variable exploration noise (SmartDrive strategy)
ACTION_STD_INIT       = 0.4
ACTION_STD_MIN        = 0.05
ACTION_STD_DECAY      = 0.05
# 4 decay steps across ~400-episode run: every 75 episodes
ACTION_STD_DECAY_FREQ = 75

# ─────────────────────────────────────────────
# REWARD WEIGHTS  (paper Table I)
# ─────────────────────────────────────────────
W_LANE   = 0.3
W_LIDAR  = 0.3
W_SPEED  = 0.2
W_CENTER = 0.2
# FIX-1: W_BLIP REMOVED. BLIP goes into state only, not reward.
# The paper (Section I-B) explicitly contrasts their approach (BLIP in state)
# with prior work that uses VLMs for reward shaping. Using both is wrong.

REWARD_TERMINAL = -10.0   # SmartDrive terminal penalty — unchanged
REWARD_FAIL     = -3.0    # soft fail (env exception)

# ─────────────────────────────────────────────
# REWARD: LIDAR TERMS  (paper Eq.15 / Table I)
# ─────────────────────────────────────────────
LIDAR_DMID         = 8.0          # m — mid-distance threshold
LIDAR_DLOW         = 4.0          # m — low-distance threshold
LIDAR_DCRIT        = 2.8          # m — critical penalty starts
LIDAR_DFAIL        = 2.0          # m — termination threshold
LIDAR_B1           = 5.0          # penalty coefficient (dmid zone)
LIDAR_B2           = 10.0         # penalty offset (dcrit zone)
LIDAR_B3           = 2.0          # penalty slope (dcrit zone)
LIDAR_BONUS        = 5.0          # bonus magnitude in safe ranges
LIDAR_BONUS_RANGES = [(3.0, 4.0), (8.0, 10.0)]   # m — safe spacing bonus

# ─────────────────────────────────────────────
# REWARD: LANE / CENTER TERMS  (paper Table I)
# ─────────────────────────────────────────────
LANE_NORM       = 100.0   # pixels — normalisation constant dlane
CENTER_MAX_DIST = 80.0    # pixels — saturation threshold dclip
CENTER_K        = 2.5     # gain k for centre penalty
REWARD_CLIP     = 2.0     # raised from paper's 1.0: W_LIDAR*LIDAR_BONUS=1.5 > 1.0 gets clipped otherwise

# Lane termination — pixel threshold (Hough detector)
# Raised from 85 → 120 px to match SmartDrive's 3.0 m threshold.
LANE_RESET_DIST = 120.0   # pixels

# Lane centre in real-world metres (SmartDrive metric)
MAX_DISTANCE_FROM_CENTER = 3.5   # metres — slightly relaxed from 3.0
                                  # The spawn waypoint geometry can start
                                  # the first step at ~3m; 3.5 gives one
                                  # step of grace before terminating.

# ─────────────────────────────────────────────
# SPEED
# ─────────────────────────────────────────────
TARGET_SPEED = 20.0   # km/h
MAX_SPEED    = 35.0
MIN_SPEED    = 5.0    # lowered from 15.0 — car needs time to accelerate from spawn
                      # 15 km/h was terminating episodes before the agent could learn
                      # to move. Raise back to 10-15 once agent consistently moves.

# ─────────────────────────────────────────────
# LIDAR TERMINATION WINDOW
# ─────────────────────────────────────────────
# Window doubled (20→40), count unchanged (10).
# Trigger = 10/40 = 25% instead of 50% — matches SmartDrive's
# more forgiving early-training termination behaviour.
LIDAR_BELOW_THRESH_COUNT = 10
LIDAR_WINDOW_STEPS       = 40

# ─────────────────────────────────────────────
# CARLA / ENVIRONMENT
# ─────────────────────────────────────────────
TOWN                 = "Town02"
CAR_NAME             = "vehicle.lincoln.mkz_2017"
NUMBER_OF_PEDESTRIAN = 0
VISUAL_DISPLAY       = True
EPISODE_LENGTH       = 10000      # max steps per episode

# ─────────────────────────────────────────────
# TRAINING / PATHS
# ─────────────────────────────────────────────
TRAIN_TIMESTEPS    = 200_000
TEST_EPISODES      = 100
CHECKPOINT_LOAD    = False
CHECKPOINT_SAVE_FREQ = 10_000    # SB3 CheckpointCallback frequency (steps)

PPO_MODEL_PATH  = "Results_BLIP/model/blip_fuseppo"
LOG_PATH_TRAIN  = "Results_BLIP/logs/train"
LOG_PATH_TEST   = "Results_BLIP/logs/test"
CHECKPOINT_PATH = "Results_BLIP/checkpoints"
RESULTS_PATH    = "Results_BLIP"