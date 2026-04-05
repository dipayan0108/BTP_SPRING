# evaluate.py
# Evaluation script — SmartDrive + BLIP-FusePPO hybrid.
#
# FIX-eval: lane_exit done_reason now uses LANE_RESET_DIST (120 px)
#   instead of hard-coded 85 px. Training and evaluation were using
#   different thresholds, making lane_exit rate metrics inconsistent.
#
# Reports comparison against:
#   - SmartDrive GPU  (mean_reward, mean_dist, mean_speed, RMSE)
#   - BLIP-FusePPO paper baselines (RMSE vs DDPG, VL-SAFE)
#   Both sets of metrics are now reported since we target both benchmarks.

import os
import sys
import time
import csv
import random
import numpy as np
from datetime import datetime

from stable_baselines3 import PPO

from parameters import (
    SEED, PPO_MODEL_PATH, RESULTS_PATH,
    EPISODE_LENGTH, TARGET_SPEED,
    LANE_RESET_DIST,   # FIX-eval: use param, not magic number
)
from environment import CarlaEnv, ClientConnection

# ── SmartDrive published results ──────────────────────────────────────
SMARTDRIVE_GPU = {
    "rmse_m":           0.0086,
    "mean_distance_m":  478.8,
    "mean_reward":      1502.50,
    "mean_speed_kmh":   17.73,
    "inference_ms":     33.01,
}
SMARTDRIVE_EDGE = {
    "mean_distance_m":  407.4,
    "mean_reward":      1236.97,
    "mean_speed_kmh":   12.92,
    "inference_ms":     36.85,
}

# ── BLIP-FusePPO paper baselines (Table III) ──────────────────────────
PAPER_BASELINES = {
    "DDPG":      {"rmse_m": 0.242, "std_m": 0.121, "nrmse": 0.0484},
    "VL-SAFE":   {"rmse_m": 0.198, "std_m": 0.099, "nrmse": 0.0396},
    "BLIP-paper":{"rmse_m": 0.110, "std_m": 0.055, "nrmse": 0.0220},
}

NUM_TEST_EPISODES = 100
SCALE_PX_TO_M     = 5.0 / 235.0    # paper's scale factor
LANE_WIDTH_M      = 5.0             # for nRMSE normalisation


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)


def compute_metrics(records):
    n = len(records)

    all_devs = []
    for r in records:
        all_devs.extend(r["lateral_deviations_m"])
    all_devs = np.array(all_devs)

    rmse     = float(np.sqrt(np.mean(all_devs ** 2)))
    std_dev  = float(np.std(all_devs))
    nrmse    = rmse / LANE_WIDTH_M
    mean_dev = float(np.mean(np.abs(all_devs)))

    distances = np.array([r["distance_covered_m"] for r in records])
    rewards   = np.array([r["total_reward"]        for r in records])
    reasons   = [r["done_reason"]                  for r in records]

    mean_dist       = float(np.mean(distances))
    mean_reward     = float(np.mean(rewards))
    completion_rate = float(sum(1 for r in reasons if r == "destination") / n)
    collision_rate  = float(sum(1 for r in reasons if r == "collision")   / n)
    lane_exit_rate  = float(sum(1 for r in reasons if r == "lane_exit")   / n)

    all_speeds = []
    for r in records:
        all_speeds.extend(r["speeds_kmh"])
    mean_speed      = float(np.mean(all_speeds)) if all_speeds else 0.0
    speed_stability = float(np.std(all_speeds))  if all_speeds else 0.0

    all_steers = []
    for r in records:
        all_steers.extend(r["steers"])
    steering_jerk = float(np.std(np.diff(all_steers))) if len(all_steers) > 1 else 0.0

    inf_times = []
    for r in records:
        inf_times.extend(r["inference_times_ms"])
    mean_inf = float(np.mean(inf_times)) if inf_times else 0.0

    return {
        "rmse_m":           rmse,
        "std_dev_m":        std_dev,
        "nrmse":            nrmse,
        "mean_distance_m":  mean_dist,
        "mean_reward":      mean_reward,
        "completion_rate":  completion_rate,
        "collision_rate":   collision_rate,
        "lane_exit_rate":   lane_exit_rate,
        "mean_speed_kmh":   mean_speed,
        "speed_stability":  speed_stability,
        "steering_jerk":    steering_jerk,
        "mean_inference_ms": mean_inf,
    }


def run_evaluation():
    set_seeds()

    try:
        client, world = ClientConnection().setup()
    except ConnectionError as e:
        print(f"CARLA refused: {e}")
        sys.exit(1)

    env = CarlaEnv(client, world)
    print(f"Loading model from {PPO_MODEL_PATH} ...")
    model = PPO.load(PPO_MODEL_PATH, env=env)
    print("Model loaded.\n")

    records = []

    for ep in range(1, NUM_TEST_EPISODES + 1):
        obs       = env.reset()
        done      = False
        ep_reward = 0.0
        devs_m    = []
        speeds    = []
        steers    = []
        inf_times = []
        done_reason = "timeout"

        for _ in range(int(EPISODE_LENGTH)):
            t0 = time.time()
            action, _ = model.predict(obs, deterministic=True)
            inf_ms = (time.time() - t0) * 1000.0
            inf_times.append(inf_ms)

            obs, reward, done, info = env.step(action)
            ep_reward += reward

            dev_px = env._last_distance_px
            devs_m.append(abs(dev_px) * SCALE_PX_TO_M)
            speeds.append(env.velocity)
            steers.append(float(action[0]))

            if done:
                # FIX-eval: use LANE_RESET_DIST from parameters (120 px)
                col_data = (
                    env.collision_obj.collision_data
                    if env.collision_obj is not None else []
                )
                if len(col_data) > 0:
                    done_reason = "collision"
                elif abs(env._last_distance_px) > LANE_RESET_DIST:
                    done_reason = "lane_exit"
                elif env.current_waypoint_index >= len(
                        env.route_waypoints or []) - 2:
                    done_reason = "destination"
                break

        dist_m = info.get("distance_covered", 0) * 1.0

        records.append({
            "lateral_deviations_m": devs_m,
            "distance_covered_m":   float(dist_m),
            "total_reward":         ep_reward,
            "speeds_kmh":           speeds,
            "steers":               steers,
            "done_reason":          done_reason,
            "inference_times_ms":   inf_times,
        })

        ep_rmse = float(np.sqrt(np.mean(np.array(devs_m)**2))) if devs_m else 0.0
        print(f"Ep {ep:3d} | Reward: {ep_reward:7.2f} | Dist: {dist_m:5.0f}m | "
              f"RMSE: {ep_rmse:.4f}m | Reason: {done_reason}")

    env.close()

    metrics = compute_metrics(records)

    # ── Save CSV ──────────────────────────────────────────────────────
    save_dir = os.path.join(RESULTS_PATH, "evaluation")
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "ours",
                         "smartdrive_gpu", "smartdrive_edge",
                         "blip_paper", "vl_safe", "ddpg"])
        rows = [
            ("rmse_m",           metrics["rmse_m"],
             SMARTDRIVE_GPU.get("rmse_m", "N/A"), "N/A",
             PAPER_BASELINES["BLIP-paper"]["rmse_m"],
             PAPER_BASELINES["VL-SAFE"]["rmse_m"],
             PAPER_BASELINES["DDPG"]["rmse_m"]),
            ("std_dev_m",        metrics["std_dev_m"],
             "N/A", "N/A",
             PAPER_BASELINES["BLIP-paper"]["std_m"],
             PAPER_BASELINES["VL-SAFE"]["std_m"],
             PAPER_BASELINES["DDPG"]["std_m"]),
            ("nrmse",            metrics["nrmse"],
             "N/A", "N/A",
             PAPER_BASELINES["BLIP-paper"]["nrmse"],
             PAPER_BASELINES["VL-SAFE"]["nrmse"],
             PAPER_BASELINES["DDPG"]["nrmse"]),
            ("mean_distance_m",  metrics["mean_distance_m"],
             SMARTDRIVE_GPU["mean_distance_m"],
             SMARTDRIVE_EDGE["mean_distance_m"],
             "N/A", "N/A", "N/A"),
            ("mean_reward",      metrics["mean_reward"],
             SMARTDRIVE_GPU["mean_reward"],
             SMARTDRIVE_EDGE["mean_reward"],
             "N/A", "N/A", "N/A"),
            ("mean_speed_kmh",   metrics["mean_speed_kmh"],
             SMARTDRIVE_GPU["mean_speed_kmh"],
             SMARTDRIVE_EDGE["mean_speed_kmh"],
             "N/A", "N/A", "N/A"),
            ("completion_rate",  metrics["completion_rate"],
             "N/A", "N/A", "N/A", "N/A", "N/A"),
            ("collision_rate",   metrics["collision_rate"],
             "N/A", "N/A", "N/A", "N/A", "N/A"),
            ("lane_exit_rate",   metrics["lane_exit_rate"],
             "N/A", "N/A", "N/A", "N/A", "N/A"),
            ("mean_inference_ms",metrics["mean_inference_ms"],
             SMARTDRIVE_GPU["inference_ms"],
             SMARTDRIVE_EDGE["inference_ms"],
             "N/A", "N/A", "N/A"),
        ]
        writer.writerows(rows)
    print(f"\nCSV saved to {csv_path}")

    # ── Print summary ─────────────────────────────────────────────────
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    lines = [
        "",
        "=" * 80,
        f"  RESULTS  ({NUM_TEST_EPISODES} test episodes)",
        "=" * 80,
        f"{'Metric':<28} {'Ours':>10} {'SD-GPU':>10} {'SD-Edge':>10} "
        f"{'BLIPpaper':>10} {'VL-SAFE':>10}",
        "-" * 80,
    ]
    table = [
        ("RMSE (m)",         metrics["rmse_m"],
         SMARTDRIVE_GPU.get("rmse_m","N/A"), "N/A",
         PAPER_BASELINES["BLIP-paper"]["rmse_m"],
         PAPER_BASELINES["VL-SAFE"]["rmse_m"]),
        ("Std Dev (m)",      metrics["std_dev_m"],
         "N/A", "N/A",
         PAPER_BASELINES["BLIP-paper"]["std_m"],
         PAPER_BASELINES["VL-SAFE"]["std_m"]),
        ("nRMSE",            metrics["nrmse"],
         "N/A", "N/A",
         PAPER_BASELINES["BLIP-paper"]["nrmse"],
         PAPER_BASELINES["VL-SAFE"]["nrmse"]),
        ("Mean distance (m)",metrics["mean_distance_m"],
         SMARTDRIVE_GPU["mean_distance_m"],
         SMARTDRIVE_EDGE["mean_distance_m"], "N/A", "N/A"),
        ("Mean reward",      metrics["mean_reward"],
         SMARTDRIVE_GPU["mean_reward"],
         SMARTDRIVE_EDGE["mean_reward"], "N/A", "N/A"),
        ("Mean speed (km/h)",metrics["mean_speed_kmh"],
         SMARTDRIVE_GPU["mean_speed_kmh"],
         SMARTDRIVE_EDGE["mean_speed_kmh"], "N/A", "N/A"),
        ("Completion rate",  metrics["completion_rate"],
         "N/A", "N/A", "N/A", "N/A"),
        ("Collision rate",   metrics["collision_rate"],
         "N/A", "N/A", "N/A", "N/A"),
        ("Lane exit rate",   metrics["lane_exit_rate"],
         "N/A", "N/A", "N/A", "N/A"),
        ("Inference (ms)",   metrics["mean_inference_ms"],
         SMARTDRIVE_GPU["inference_ms"],
         SMARTDRIVE_EDGE["inference_ms"], "N/A", "N/A"),
    ]
    for row in table:
        name = row[0]
        vals = [fmt(v) for v in row[1:]]
        lines.append(f"{name:<28} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} "
                     f"{vals[3]:>10} {vals[4]:>10}")
    lines.append("=" * 80)

    # RMSE improvements
    if isinstance(SMARTDRIVE_GPU.get("rmse_m"), float):
        imp = (SMARTDRIVE_GPU["rmse_m"] - metrics["rmse_m"]) \
              / SMARTDRIVE_GPU["rmse_m"] * 100
        lines.append(f"\nRMSE vs SmartDrive GPU:   {imp:+.1f}% "
                     f"({'better' if imp > 0 else 'worse'})")
    imp_paper = (PAPER_BASELINES["BLIP-paper"]["rmse_m"] - metrics["rmse_m"]) \
                / PAPER_BASELINES["BLIP-paper"]["rmse_m"] * 100
    lines.append(f"RMSE vs BLIP paper target: {imp_paper:+.1f}% "
                 f"({'better' if imp_paper > 0 else 'worse'})")
    imp_vl = (PAPER_BASELINES["VL-SAFE"]["rmse_m"] - metrics["rmse_m"]) \
             / PAPER_BASELINES["VL-SAFE"]["rmse_m"] * 100
    lines.append(f"RMSE vs VL-SAFE:           {imp_vl:+.1f}% "
                 f"({'better' if imp_vl > 0 else 'worse'})")

    dist_imp = (metrics["mean_distance_m"] - SMARTDRIVE_GPU["mean_distance_m"]) \
               / SMARTDRIVE_GPU["mean_distance_m"] * 100
    lines.append(f"Distance vs SmartDrive GPU:{dist_imp:+.1f}% "
                 f"({'better' if dist_imp > 0 else 'worse'})")

    summary = "\n".join(lines)
    print(summary)

    summary_path = os.path.join(save_dir, "metrics_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"\nSummary saved to {summary_path}")

    return metrics


if __name__ == "__main__":
    run_evaluation()