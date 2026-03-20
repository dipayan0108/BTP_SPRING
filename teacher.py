# teacher.py
# =============================================================================
#
#  Qwen-VL-Chat Teacher Model  —  SmartDrive BTP
#
#  ROLE IN THE BTP PIPELINE:
#  ─────────────────────────────────────────────────────────────────────────
#  This is the TEACHER in the Teacher-Student architecture (PDF Fig 1).
#  It is a multimodal LLM (Qwen-VL-Chat) that:
#    1. Receives the raw front-camera image + full vehicle state
#    2. Reasons about the scene in natural language
#    3. Outputs an expert driving action
#
#  The teacher is used ONLY during data collection (Step 2 in the PDF).
#  At inference time the teacher is discarded — only the student runs.
#
#  WHAT THIS FILE PROVIDES:
#  ─────────────────────────────────────────────────────────────────────────
#  • TeacherModel class  — loads Qwen-VL-Chat, runs inference
#  • get_action()        — main entry point called by collect_distillation_data.py
#  • ACTION_TO_CONTINUOUS — maps discrete labels to (steer, throttle_raw)
#                           which plug directly into CARLA env.step()
#
#  INPUTS TO get_action() — all 5 PDF vehicle state inputs + image + extras:
#    image_obs            : 160×80 BGR semantic-segmentation frame (numpy)
#    velocity             : speed in km/h
#    distance_from_center : metres from lane centre
#    angle                : heading error in radians  (from angle_diff())
#    steering_angle       : current wheel position [-1, +1]  (previous_steer)
#    throttle             : current throttle [0, 1]          (self.throttle)
#    nav_command          : 0=straight 1=left 2=right 3=follow
#    collision_occurred   : True if collision_data non-empty this timestep
#
#  OUTPUTS — the distillation labels:
#    action_label      : str    e.g. "steer_left_slight"
#    continuous_action : tuple  (steer, throttle_raw) → env.step()
#    reason            : str    LLM natural-language justification
#
#  HOW TO RUN STANDALONE TEST:
#    python teacher.py
#
# =============================================================================

import re
import json
import logging
import numpy as np
from PIL import Image

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level  = logging.INFO,
    format = '[%(asctime)s] [Teacher] %(message)s',
    datefmt= '%H:%M:%S',
)
logger = logging.getLogger('TeacherModel')

# =============================================================================
# ACTION SPACE
# =============================================================================
#
#  WHY DISCRETE LABELS:
#  Qwen-VL is a language model — it reasons in words, not floats.
#  We give it 8 named actions to choose from, then map each label to a
#  precise continuous (steer, throttle_raw) pair that CARLA understands.
#
#  ACTION SPACE DESIGN — aligned with main.py reward function:
#    target_speed = 22 km/h   min_speed = 15 km/h   max_speed = 35 km/h
#    max_distance_from_center = 3.0 m
#
#  throttle_raw ∈ [-1, 1] → actual_throttle = (throttle_raw + 1) / 2
#    throttle_raw  0.6  → actual 0.80  comfortable cruise  (~20-22 km/h)
#    throttle_raw  0.5  → actual 0.75  moderate  (during turns)
#    throttle_raw  0.3  → actual 0.65  slow  (hard correction needed)
#    throttle_raw  1.0  → actual 1.00  full power  (only if speed < 15)
#    throttle_raw  0.0  → actual 0.50  coast  (gently reduce speed)
#    throttle_raw -1.0  → actual 0.00  full brake
#
# =============================================================================

ACTION_LABELS = [
    "go_straight",
    "steer_left_slight",
    "steer_left_hard",
    "steer_right_slight",
    "steer_right_hard",
    "accelerate",
    "decelerate",
    "brake",
]

#                                  steer    throttle_raw
ACTION_TO_CONTINUOUS = {
    "go_straight":        (  0.0,   0.6),   # centre lane, comfortable cruise
    "steer_left_slight":  ( -0.2,   0.5),   # < 0.5 m right of centre
    "steer_left_hard":    ( -0.5,   0.3),   # ≥ 0.5 m right of centre, slow down
    "steer_right_slight": (  0.2,   0.5),   # < 0.5 m left of centre
    "steer_right_hard":   (  0.5,   0.3),   # ≥ 0.5 m left of centre, slow down
    "accelerate":         (  0.0,   1.0),   # speed critically below 15 km/h
    "decelerate":         (  0.0,   0.0),   # speed above 22 km/h, coast down
    "brake":              (  0.0,  -1.0),   # immediate hazard or collision
}

# =============================================================================
# CONSTANTS
# =============================================================================

MODEL_NAME      = "Qwen/Qwen-VL-Chat"
TEMP_IMAGE_PATH = "/tmp/qwenvl_teacher_frame.png"

# CARLA reward thresholds — must match main.py exactly
TARGET_SPEED   = 22.0   # km/h
MIN_SPEED      = 15.0   # km/h
MAX_SPEED      = 35.0   # km/h
MAX_DIST_CENTRE = 3.0   # metres

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def save_temp_image(image_array: np.ndarray, path: str) -> str:
    """
    Save a CARLA semantic-segmentation frame to disk for Qwen-VL.

    CARLA's CityScapesPalette callback returns pixels in BGR order
    (OpenCV convention).  PIL's Image.fromarray() expects RGB.
    We swap channels here so Qwen-VL sees correct colours:
      road  = dark grey/purple  (not orange)
      sky   = light blue        (not light orange)
      vehicles = red/orange     (not blue/cyan)
    """
    if image_array.ndim == 3 and image_array.shape[2] == 3:
        image_array = image_array[:, :, ::-1].copy()   # BGR → RGB
    Image.fromarray(image_array.astype(np.uint8)).save(path)
    return path


def nav_command_to_text(nav_command: int) -> str:
    """Convert integer navigation command to plain English for the prompt."""
    return {
        0: "go straight ahead",
        1: "prepare to turn left at the next junction",
        2: "prepare to turn right at the next junction",
        3: "follow the current lane",
    }.get(int(nav_command), "follow the current lane")


def parse_action_from_response(response_text: str) -> str:
    """
    Extract the action label from the model's text output.

    Priority order:
      1. JSON parse  — looks for {"action": "...", ...}
      2. Exact label scan  — finds label string anywhere in text
      3. Space-separated scan  — handles "steer left slight" → "steer_left_slight"
      4. Fallback  — go_straight (safe default)
    """
    # 1. JSON parse — try every {...} block in the response
    for match in re.finditer(r'\{[^{}]*\}', response_text, re.DOTALL):
        try:
            data   = json.loads(match.group())
            action = str(data.get("action", "")).strip().lower().replace(" ", "_")
            if action in ACTION_LABELS:
                return action
        except (json.JSONDecodeError, AttributeError):
            continue

    # 2. Exact label scan (with underscores)
    lower = response_text.lower()
    for label in ACTION_LABELS:
        if label in lower:
            return label

    # 3. Space-separated scan
    for label in ACTION_LABELS:
        if label.replace("_", " ") in lower:
            return label

    # 4. Safe fallback
    logger.warning(f"Cannot parse action from: '{response_text[:100]}' → go_straight")
    return "go_straight"


def extract_reason(response_text: str) -> str:
    """
    Extract the reason sentence from JSON response.
    New prompt includes 'think' field — log it at DEBUG, return only 'reason'.
    """
    for match in re.finditer(r'\{[^{}]*\}', response_text, re.DOTALL):
        try:
            data = json.loads(match.group())
            if "think" in data:
                logger.debug(f"[Teacher think] {data['think']}")
            if "reason" in data:
                return str(data["reason"]).strip()
        except (json.JSONDecodeError, AttributeError):
            continue
    return response_text.strip()[:200]

# =============================================================================
# TEACHER MODEL
# =============================================================================

class TeacherModel:
    """
    Qwen-VL-Chat based expert teacher for SmartDrive BTP.

    Implements the Teacher stage from PDF Fig 1:
      RGB image + vehicle state → scene reasoning → expert action

    The teacher drives the CARLA vehicle during data collection and
    simultaneously labels every frame.  It is discarded at deployment.

    Usage:
        teacher = TeacherModel()

        action_label, (steer, throttle_raw), reason = teacher.get_action(
            image_obs            = frame,        # (80, 160, 3) BGR uint8
            velocity             = 20.5,         # km/h
            distance_from_center = 0.3,          # metres
            angle                = 0.05,         # radians
            steering_angle       = 0.01,         # current wheel [-1,+1]
            throttle             = 0.55,         # current throttle [0,1]
            nav_command          = 0,            # 0=straight
            collision_occurred   = False,
        )
    """

    def __init__(
        self,
        model_name:      str  = MODEL_NAME,
        device:          str  = "cuda",
        torch_dtype            = torch.float16,
        temp_image_path: str  = TEMP_IMAGE_PATH,
    ):
        self.temp_image_path = temp_image_path

        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)

        logger.info(f"Loading model → {device} | {torch_dtype}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map        = device,
            torch_dtype       = torch_dtype,
            trust_remote_code = True,
        ).eval()

        # Greedy decoding — deterministic, consistent labels across frames
        self.model.generation_config = GenerationConfig.from_pretrained(
            model_name,
            trust_remote_code = True,
            do_sample         = False,
            temperature       = None,
            top_p             = None,
            max_new_tokens    = 200,   # think field needs more tokens than action+reason alone
        )

        logger.info("Teacher model ready ✓")

    # ─────────────────────────────────────────────────────────────────────────
    # PROMPT BUILDER
    # ─────────────────────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        image_path:           str,
        velocity:             float,
        nav_command:          int,
        distance_from_center: float,
        angle_deg:            float,
        steering_angle:       float,
        throttle:             float,
        collision_occurred:   bool,
    ) -> str:
        """
        Build the complete Qwen-VL-Chat query.

        PROMPT DESIGN DECISIONS:
        ─────────────────────────────────────────────────────────────────────
        1. IMAGE FIRST — Qwen-VL embeds the image token at the START of the
           query via tokenizer.from_list_format(). The text instruction
           follows immediately after.

        2. SEMANTIC COLOUR LEGEND — The CARLA semantic-segmentation camera
           produces unusual colours. Qwen-VL was pretrained on normal photos
           and needs to be told what each colour means in this context.

        3. ALL 5 PDF VEHICLE STATE INPUTS are shown explicitly in natural
           language so the model can reason about them:
             • velocity           → "Current speed: 20.5 km/h"
             • steering_angle     → "Steering wheel: turned LEFT (-0.45)"
             • throttle           → "Throttle: high (0.75)"
             • distance_from_center → "Distance from centre: 1.20 m"
             • heading error      → "Heading error: 11.5 degrees right"

        4. COLLISION IS RULE 0 — shown as a warning banner at top AND as the
           first decision rule. The model must output "brake" immediately.
           We also enforce this in Python (hard override) as a safety net.

        5. CONTINUITY NOTES — teacher knows current wheel angle and throttle
           so it can avoid overcorrection (don't steer hard if already
           steering hard in that direction).

        6. ONE CONCRETE EXAMPLE — few-shot examples dramatically improve
           JSON compliance in vision-language models.

        7. NO "obey traffic lights" — we override red lights in the CARLA
           environment. Including this instruction would contradict the
           behaviour the student observes.
        """

        nav_text = nav_command_to_text(nav_command)

        # ── human-readable steering state ─────────────────────────────────────
        if abs(steering_angle) < 0.05:
            steer_desc = f"straight  ({steering_angle:+.2f})"
        elif steering_angle < 0:
            steer_desc = f"turned LEFT  ({steering_angle:+.2f})"
        else:
            steer_desc = f"turned RIGHT  ({steering_angle:+.2f})"

        # ── human-readable throttle state ─────────────────────────────────────
        if throttle < 0.2:
            throttle_desc = f"very low  ({throttle:.2f})  — nearly coasting"
        elif throttle < 0.5:
            throttle_desc = f"moderate  ({throttle:.2f})"
        elif throttle < 0.8:
            throttle_desc = f"high  ({throttle:.2f})"
        else:
            throttle_desc = f"full  ({throttle:.2f})  — maximum power"

        # ── collision warning banner ───────────────────────────────────────────
        collision_banner = (
            "╔══════════════════════════════════════════════╗\n"
            "║  ⚠  COLLISION DETECTED — output brake NOW  ║\n"
            "╚══════════════════════════════════════════════╝\n\n"
        ) if collision_occurred else ""

        # ── full vehicle state block ───────────────────────────────────────────
        state_block = (
            f"  Speed            : {velocity:.1f} km/h"
            f"  (target 15–22 km/h, max 35 km/h)\n"
            f"  Steering wheel   : {steer_desc}"
            f"  (scale: -1.0=hard left, 0.0=straight, +1.0=hard right)\n"
            f"  Throttle         : {throttle_desc}"
            f"  (scale: 0.0=no power, 1.0=full power)\n"
            f"  Distance centre  : {distance_from_center:.2f} m"
            f"  (0.0=perfectly centred, 3.0=edge of allowed range)\n"
            f"  Heading error    : {abs(angle_deg):.1f} degrees "
            f"{'LEFT' if angle_deg < 0 else 'RIGHT'} of lane direction\n"
            f"  Navigation       : {nav_text}"
        )

        # ── full instruction ───────────────────────────────────────────────────
        instruction = (
            f"{collision_banner}"

            # ── ROLE + CONTEXT ────────────────────────────────────────────────
            "You are an expert driving instructor observing a vehicle in a CARLA urban simulation.\n"
            "The vehicle is driven by an autopilot. Your job is to LABEL each frame — not to drive.\n"
            "Look at the image and vehicle state, then decide what the CORRECT action should be.\n"
            "Your labels will be used to train a student network via imitation learning.\n"
            "Every label must be the SAFEST and MOST CORRECT action for this situation.\n\n"

            # ── FIX A: explicit image analysis step ───────────────────────────
            "═══ STEP 1 — ANALYSE THE IMAGE ═══\n"
            "Before deciding, examine the front-camera image carefully.\n"
            "The image uses SEMANTIC SEGMENTATION colours (not real RGB):\n"
            "  Dark grey / purple  = road surface\n"
            "  White dashes        = lane markings\n"
            "  Red / orange        = other vehicles\n"
            "  Dark green          = vegetation / buildings\n"
            "  Light blue          = sky\n"
            "  Pink / magenta      = pedestrians\n"
            "Ask yourself: Is the road ahead clear? Are there any red/orange or pink\n"
            "pixels directly in the centre of the image (in the vehicle's path)?\n"
            "Is the lane visible? Is the vehicle centred between the white dashes?\n\n"

            # ── VEHICLE STATE ─────────────────────────────────────────────────
            "═══ STEP 2 — READ VEHICLE STATE ═══\n"
            f"{state_block}\n\n"

            # ── FIX B: explicit direction convention ──────────────────────────
            "═══ STEP 3 — DIRECTION CONVENTION (read carefully) ═══\n"
            "  'distance from centre' means how far the car has drifted from the lane middle.\n"
            "  If the car has drifted to the RIGHT  → the correction is to steer LEFT.\n"
            "  If the car has drifted to the LEFT   → the correction is to steer RIGHT.\n"
            "  Heading error RIGHT  means the car's nose points right of the lane → steer LEFT.\n"
            "  Heading error LEFT   means the car's nose points left of the lane  → steer RIGHT.\n\n"

            # ── AVAILABLE ACTIONS ─────────────────────────────────────────────
            "═══ STEP 4 — CHOOSE ONE ACTION ═══\n"
            "  go_straight         road clear, speed 15–22 km/h, vehicle centred (< 0.5 m)\n"
            "  steer_left_slight   car drifted RIGHT 0–0.5 m  OR  nose points right < 10 deg\n"
            "  steer_left_hard     car drifted RIGHT > 0.5 m  OR  nose points right > 10 deg\n"
            "  steer_right_slight  car drifted LEFT  0–0.5 m  OR  nose points left  < 10 deg\n"
            "  steer_right_hard    car drifted LEFT  > 0.5 m  OR  nose points left  > 10 deg\n"
            "  accelerate          speed < 15 km/h  AND  no hazard visible ahead\n"
            "  decelerate          speed > 22 km/h  OR   slow obstacle visible ahead\n"
            "  brake               hazard directly ahead (red/orange/pink pixels in centre)\n\n"

            # ── DECISION RULES ────────────────────────────────────────────────
            "═══ DECISION RULES (strict priority order) ═══\n"
            "  RULE 0 [EMERGENCY]: Collision detected → output brake. Always.\n"
            "  RULE 1 [SAFETY]:    Pedestrian (pink) or vehicle (red/orange) in path → brake.\n"
            "  RULE 2 [LANE]:      Distance from centre > 0.5 m → steer to correct.\n"
            "    Correction direction: drifted right → steer LEFT. Drifted left → steer RIGHT.\n"
            "    Wheel already correcting hard in right direction → choose slight over hard.\n"
            "    Distance > 1.5 m → always choose hard correction.\n"
            "  RULE 3 [SPEED LOW]: Speed < 15 km/h and road clear → accelerate.\n"
            "    Throttle already full (≥ 0.8) → choose go_straight instead.\n"
            "  RULE 4 [SPEED HIGH]: Speed > 22 km/h → decelerate.\n"
            "  RULE 5 [DEFAULT]:   All conditions normal → go_straight.\n\n"

            # ── FIX C: 3 diverse few-shot examples ───────────────────────────
            "═══ EXAMPLES ═══\n"

            "Example 1 — Normal cruising:\n"
            "  State: speed=20 km/h, dist=0.1 m, heading=2 deg right, steer=straight, throttle=moderate\n"
            "  Image: clear road, no obstacles, vehicle centred between lane markings\n"
            '  Output: {"think": "Road clear, speed in range, centred. No action needed.", '
            '"action": "go_straight", "reason": "Road clear and vehicle centred at target speed."}\n\n'

            "Example 2 — Drifted right, needs left correction:\n"
            "  State: speed=18 km/h, dist=1.2 m RIGHT of centre, heading=11 deg right, steer=slight left\n"
            "  Image: lane markings visible, car is near right edge\n"
            '  Output: {"think": "Drifted 1.2 m right. Correction is steer LEFT. '
            'Wheel slightly left already so use slight not hard.", '
            '"action": "steer_left_slight", "reason": "Drifted right of centre, applying gentle left correction."}\n\n'

            "Example 3 — Vehicle ahead, must brake:\n"
            "  State: speed=24 km/h, dist=0.2 m, heading=1 deg, steer=straight, throttle=high\n"
            "  Image: large red/orange block in centre-bottom of image (vehicle directly ahead)\n"
            '  Output: {"think": "Red vehicle visible directly ahead in centre of image. '
            'Rule 1 applies. Must brake immediately.", '
            '"action": "brake", "reason": "Vehicle detected directly ahead, emergency braking."}\n\n'

            # ── FIX D: think-first output format ─────────────────────────────
            "═══ YOUR OUTPUT ═══\n"
            "Respond with ONLY valid JSON. No text before or after the JSON.\n"
            "The 'think' field is your private reasoning — use it to apply the rules above.\n"
            "The 'action' field must be EXACTLY one of the 8 labels above.\n"
            "The 'reason' field is one sentence (max 15 words) explaining your choice.\n"
            '{"think": "<apply rules step by step>", '
            '"action": "<exactly one of the 8 labels>", '
            '"reason": "<one sentence max 15 words>"}'
        )

        return self.tokenizer.from_list_format([
            {"image": image_path},
            {"text":  instruction},
        ])

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN INFERENCE ENTRY POINT
    # ─────────────────────────────────────────────────────────────────────────

    def get_action(
        self,
        image_obs:            np.ndarray,
        velocity:             float,
        distance_from_center: float  = 0.0,
        angle:                float  = 0.0,
        steering_angle:       float  = 0.0,
        throttle:             float  = 0.0,
        nav_command:          int    = 0,
        collision_occurred:   bool   = False,
    ) -> tuple:
        """
        Run one teacher inference step.

        Args:
            image_obs            np.ndarray   (H, W, 3) BGR uint8
                                              from CARLA CityScapesPalette camera
            velocity             float        speed in km/h
            distance_from_center float        metres from lane centre
                                              from main.py distance_to_line()
            angle                float        heading error in RADIANS
                                              from main.py angle_diff()
            steering_angle       float        current wheel angle [-1, +1]
                                              = previous_steer from main.py
            throttle             float        current throttle [0, 1]
                                              = self.throttle from main.py
            nav_command          int          0=straight 1=left 2=right 3=follow
            collision_occurred   bool         True if collision_data non-empty

        Returns:
            action_label      str    one of the 8 ACTION_LABELS
            continuous_action tuple  (steer, throttle_raw) for env.step()
            reason            str    LLM justification sentence
        """

        # ── HARD SAFETY OVERRIDE ──────────────────────────────────────────────
        # If collision already detected, bypass LLM entirely.
        # The LLM is instructed to output brake too, but this Python-level
        # override guarantees correctness even if the model ignores Rule 0.
        if collision_occurred:
            logger.info("COLLISION OVERRIDE → brake (bypassed LLM)")
            return "brake", ACTION_TO_CONTINUOUS["brake"], "Collision detected, emergency stop."

        # ── 1. Save image to disk (BGR → RGB swap happens inside) ─────────────
        image_path = save_temp_image(image_obs, self.temp_image_path)

        # ── 2. Convert angle: radians → degrees for the prompt ────────────────
        angle_deg = float(np.degrees(angle))

        # ── 3. Build prompt ───────────────────────────────────────────────────
        query = self._build_prompt(
            image_path           = image_path,
            velocity             = float(velocity),
            nav_command          = int(nav_command),
            distance_from_center = float(distance_from_center),
            angle_deg            = angle_deg,
            steering_angle       = float(steering_angle),
            throttle             = float(throttle),
            collision_occurred   = bool(collision_occurred),
        )

        # ── 4. Run Qwen-VL-Chat inference ─────────────────────────────────────
        with torch.no_grad():
            response, _ = self.model.chat(
                self.tokenizer,
                query   = query,
                history = None,    # single-turn, no chat history
            )

        logger.debug(f"Raw response: {response}")

        # ── 5. Parse action and reason ────────────────────────────────────────
        action_label      = parse_action_from_response(response)
        continuous_action = ACTION_TO_CONTINUOUS[action_label]
        reason            = extract_reason(response)

        logger.info(
            f"label={action_label:<22} | "
            f"steer={continuous_action[0]:+.2f} | "
            f"thr_raw={continuous_action[1]:+.2f} | "
            f"reason={reason[:70]}")

        return action_label, continuous_action, reason


# =============================================================================
# STANDALONE SANITY TEST
# Run:  python teacher.py
# =============================================================================

if __name__ == "__main__":
    import time

    print("=" * 65)
    print("  TeacherModel — Qwen-VL-Chat  |  Sanity Test")
    print("  First run downloads ~10 GB from HuggingFace")
    print("=" * 65)

    # Structured synthetic image that looks like a real CARLA semantic frame
    # (80×160 BGR — same spec as the camera in main.py)
    frame = np.zeros((80, 160, 3), dtype=np.uint8)
    frame[:35]          = [180, 130,  70]   # sky  (BGR light blue)
    frame[35:]          = [ 81,   0,  81]   # road (BGR dark purple)
    frame[68:74, 40:60] = [  0, 165, 255]   # orange vehicle ahead (BGR)
    frame[70:75, 50:52] = [255, 182, 193]   # pedestrian pixel (BGR pink)

    teacher = TeacherModel()

    tests = [
        dict(
            name                 = "Normal cruise — centred at target speed",
            image_obs            = frame,
            velocity             = 20.0,
            distance_from_center = 0.1,
            angle                = 0.05,
            steering_angle       = 0.01,
            throttle             = 0.55,
            nav_command          = 0,
            collision_occurred   = False,
            expected_hint        = "go_straight or decelerate",
        ),
        dict(
            name                 = "Drifted RIGHT — wheel already correcting left",
            image_obs            = frame,
            velocity             = 18.0,
            distance_from_center = 1.2,
            angle                = 0.2,
            steering_angle       = -0.45,   # already hard left
            throttle             = 0.50,
            nav_command          = 0,
            collision_occurred   = False,
            expected_hint        = "steer_left_slight  (NOT hard — wheel already left)",
        ),
        dict(
            name                 = "Speed too low — road clear",
            image_obs            = frame,
            velocity             = 10.0,
            distance_from_center = 0.2,
            angle                = 0.0,
            steering_angle       = 0.0,
            throttle             = 0.1,
            nav_command          = 0,
            collision_occurred   = False,
            expected_hint        = "accelerate",
        ),
        dict(
            name                 = "Collision just occurred — MUST be brake",
            image_obs            = frame,
            velocity             = 22.0,
            distance_from_center = 0.3,
            angle                = 0.1,
            steering_angle       = 0.1,
            throttle             = 0.6,
            nav_command          = 0,
            collision_occurred   = True,
            expected_hint        = "brake  [enforced by Python override]",
        ),
    ]

    for i, t in enumerate(tests, 1):
        name = t.pop("name")
        hint = t.pop("expected_hint")
        print(f"\n── Test {i}: {name}")
        print(f"   Expected hint : {hint}")
        t0 = time.time()
        label, action, reason = teacher.get_action(**t)
        print(f"   label         : {label}")
        print(f"   steer         : {action[0]:+.2f}")
        print(f"   throttle_raw  : {action[1]:+.2f}")
        print(f"   reason        : {reason}")
        print(f"   time          : {time.time()-t0:.2f}s")
        if t.get("collision_occurred"):
            assert label == "brake", f"FAIL — collision must force brake, got {label}"
            print("   PASS ✓")

    print("\n" + "=" * 65)
    print("  All tests complete ✓")
    print("=" * 65)