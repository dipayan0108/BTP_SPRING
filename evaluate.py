# evaluate.py
# Computes all comparison metrics between SmartDrive and our system.
# Run after training:  python evaluate.py
#
# Outputs:
#   - Console summary table
#   - Results_BLIP/evaluation/metrics.csv
#   - Results_BLIP/evaluation/metrics_summary.txt

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
)
from environment import CarlaEnv, ClientConnection

# ── SmartDrive published results for comparison ─────────────────────
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
    "loop_latency_ms":  118.45,
}

NUM_TEST_EPISODES  = 100   # match BLIP-FusePPO evaluation protocol
SCALE_PX_TO_M      = 5.0 / 235.0   # 0.02128 m/px
LANE_WIDTH_M       = 5.0


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)


def compute_metrics(records):
    """
    records: list of dicts, one per episode, each containing:
        lateral_deviations_m : list of float  (per timestep)
        distance_covered_m   : float
        total_reward         : float
        speeds_kmh           : list of float
        steers               : list of float
        done_reason          : str  (collision/lane_exit/destination/timeout)
        inference_times_ms   : list of float
    """
    n = len(records)

    # ── Lateral deviation metrics ─────────────────────────────────
    all_devs = []
    for r in records:
        all_devs.extend(r["lateral_deviations_m"])
    all_devs = np.array(all_devs)

    rmse    = float(np.sqrt(np.mean(all_devs ** 2)))
    std_dev = float(np.std(all_devs))
    nrmse   = rmse / LANE_WIDTH_M
    mean_dev = float(np.mean(np.abs(all_devs)))

    # ── Episode-level metrics ─────────────────────────────────────
    distances = np.array([r["distance_covered_m"] for r in records])
    rewards   = np.array([r["total_reward"]        for r in records])
    reasons   = [r["done_reason"]                  for r in records]

    mean_dist       = float(np.mean(distances))
    mean_reward     = float(np.mean(rewards))
    completion_rate = float(sum(1 for r in reasons if r == "destination") / n)
    collision_rate  = float(sum(1 for r in reasons if r == "collision")   / n)
    lane_exit_rate  = float(sum(1 for r in reasons if r == "lane_exit")   / n)

    # ── Speed metrics ─────────────────────────────────────────────
    all_speeds = []
    for r in records:
        all_speeds.extend(r["speeds_kmh"])
    mean_speed      = float(np.mean(all_speeds))
    speed_stability = float(np.mean(np.abs(
        np.array(all_speeds) - TARGET_SPEED)))

    # ── Steering smoothness ───────────────────────────────────────
    all_jerk = []
    for r in records:
        steers = r["steers"]
        if len(steers) > 1:
            jerk = np.mean(np.abs(np.diff(steers)))
            all_jerk.append(jerk)
    steering_jerk = float(np.mean(all_jerk)) if all_jerk else 0.0

    # ── Inference time ────────────────────────────────────────────
    all_inf = []
    for r in records:
        all_inf.extend(r["inference_times_ms"])
    mean_inference_ms = float(np.mean(all_inf)) if all_inf else 0.0

    return {
        "rmse_m":            rmse,
        "std_dev_m":         std_dev,
        "nrmse":             nrmse,
        "mean_deviation_m":  mean_dev,
        "mean_distance_m":   mean_dist,
        "mean_reward":       mean_reward,
        "completion_rate":   completion_rate,
        "collision_rate":    collision_rate,
        "lane_exit_rate":    lane_exit_rate,
        "mean_speed_kmh":    mean_speed,
        "speed_stability":   speed_stability,
        "steering_jerk":     steering_jerk,
        "mean_inference_ms": mean_inference_ms,
        "num_episodes":      n,
    }


def run_evaluation():
    set_seeds()

    print("\n" + "="*60)
    print("  EVALUATION — {} episodes".format(NUM_TEST_EPISODES))
    print("="*60 + "\n")

    # ── Connect to CARLA ──────────────────────────────────────────
    try:
        client, world = ClientConnection().setup()
    except ConnectionError as e:
        print("CARLA connection failed: {}".format(e))
        sys.exit(1)

    env   = CarlaEnv(client, world)
    model = PPO.load(PPO_MODEL_PATH, env=env)
    print("Model loaded from {}\n".format(PPO_MODEL_PATH))

    # ── Run episodes ──────────────────────────────────────────────
    records = []

    for ep in range(1, NUM_TEST_EPISODES + 1):
        obs         = env.reset()
        done        = False
        ep_reward   = 0.0
        devs_m      = []
        speeds      = []
        steers      = []
        inf_times   = []
        done_reason = "timeout"

        for _ in range(int(EPISODE_LENGTH)):
            t0 = time.time()
            action, _ = model.predict(obs, deterministic=True)
            inf_ms = (time.time() - t0) * 1000.0
            inf_times.append(inf_ms)

            obs, reward, done, info = env.step(action)
            ep_reward += reward

            # Lateral deviation in metres
            dev_px = env._last_distance_px
            devs_m.append(abs(dev_px) * SCALE_PX_TO_M)

            # Speed and steering
            speeds.append(env.velocity)
            steers.append(float(action[0]))

            if done:
                # Determine termination reason
                if len(env.collision_obj.collision_data
                       if env.collision_obj else []) > 0:
                    done_reason = "collision"
                elif abs(dev_px) > 85:
                    done_reason = "lane_exit"
                elif env.current_waypoint_index >= len(
                        env.route_waypoints or []) - 2:
                    done_reason = "destination"
                break

        dist_m = info.get("distance_covered", 0) * 1.0  # already metres
        # Convert waypoint count to metres (1 waypoint = 1m spacing)

        records.append({
            "lateral_deviations_m": devs_m,
            "distance_covered_m":   float(dist_m),
            "total_reward":         ep_reward,
            "speeds_kmh":           speeds,
            "steers":               steers,
            "done_reason":          done_reason,
            "inference_times_ms":   inf_times,
        })

        print("Ep {:3d} | Reward: {:7.2f} | Dist: {:5.0f}m | "
              "RMSE: {:.4f}m | Reason: {}".format(
                  ep, ep_reward, dist_m,
                  float(np.sqrt(np.mean(np.array(devs_m)**2))),
                  done_reason))

    env.close()

    # ── Compute metrics ───────────────────────────────────────────
    metrics = compute_metrics(records)

    # ── Save to CSV ───────────────────────────────────────────────
    save_dir = os.path.join(RESULTS_PATH, "evaluation")
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "our_system",
                         "smartdrive_gpu", "smartdrive_edge"])
        rows = [
            ("rmse_m",           metrics["rmse_m"],
             SMARTDRIVE_GPU.get("rmse_m", "N/A"),
             "N/A"),
            ("std_dev_m",        metrics["std_dev_m"],       "N/A", "N/A"),
            ("nrmse",            metrics["nrmse"],           "N/A", "N/A"),
            ("mean_distance_m",  metrics["mean_distance_m"],
             SMARTDRIVE_GPU["mean_distance_m"],
             SMARTDRIVE_EDGE["mean_distance_m"]),
            ("mean_reward",      metrics["mean_reward"],
             SMARTDRIVE_GPU["mean_reward"],
             SMARTDRIVE_EDGE["mean_reward"]),
            ("completion_rate",  metrics["completion_rate"], "N/A", "N/A"),
            ("collision_rate",   metrics["collision_rate"],  "N/A", "N/A"),
            ("lane_exit_rate",   metrics["lane_exit_rate"],  "N/A", "N/A"),
            ("mean_speed_kmh",   metrics["mean_speed_kmh"],
             SMARTDRIVE_GPU["mean_speed_kmh"],
             SMARTDRIVE_EDGE["mean_speed_kmh"]),
            ("speed_stability",  metrics["speed_stability"], "N/A", "N/A"),
            ("steering_jerk",    metrics["steering_jerk"],   "N/A", "N/A"),
            ("mean_inference_ms",metrics["mean_inference_ms"],
             SMARTDRIVE_GPU["inference_ms"],
             SMARTDRIVE_EDGE["inference_ms"]),
        ]
        writer.writerows(rows)
    print("\nCSV saved to {}".format(csv_path))

    # ── Print summary table ───────────────────────────────────────
    summary_path = os.path.join(save_dir, "metrics_summary.txt")
    lines = []
    lines.append("\n" + "="*70)
    lines.append("  COMPARISON RESULTS  ({} test episodes)".format(
        NUM_TEST_EPISODES))
    lines.append("="*70)
    lines.append("{:<28} {:>12} {:>14} {:>12}".format(
        "Metric", "Ours", "SmartDrive-GPU", "SmartDrive-Edge"))
    lines.append("-"*70)

    def fmt(val):
        if isinstance(val, float):
            return "{:.4f}".format(val)
        return str(val)

    table = [
        ("RMSE (m)",          metrics["rmse_m"],
         SMARTDRIVE_GPU.get("rmse_m","N/A"), "N/A"),
        ("Std Dev (m)",       metrics["std_dev_m"],       "N/A", "N/A"),
        ("nRMSE",             metrics["nrmse"],           "N/A", "N/A"),
        ("Mean distance (m)", metrics["mean_distance_m"],
         SMARTDRIVE_GPU["mean_distance_m"],
         SMARTDRIVE_EDGE["mean_distance_m"]),
        ("Mean reward",       metrics["mean_reward"],
         SMARTDRIVE_GPU["mean_reward"],
         SMARTDRIVE_EDGE["mean_reward"]),
        ("Completion rate",   metrics["completion_rate"], "N/A", "N/A"),
        ("Collision rate",    metrics["collision_rate"],  "N/A", "N/A"),
        ("Lane exit rate",    metrics["lane_exit_rate"],  "N/A", "N/A"),
        ("Mean speed (km/h)", metrics["mean_speed_kmh"],
         SMARTDRIVE_GPU["mean_speed_kmh"],
         SMARTDRIVE_EDGE["mean_speed_kmh"]),
        ("Speed stability",   metrics["speed_stability"], "N/A", "N/A"),
        ("Steering jerk",     metrics["steering_jerk"],   "N/A", "N/A"),
        ("Inference (ms)",    metrics["mean_inference_ms"],
         SMARTDRIVE_GPU["inference_ms"],
         SMARTDRIVE_EDGE["inference_ms"]),
    ]

    for name, ours, sd_gpu, sd_edge in table:
        lines.append("{:<28} {:>12} {:>14} {:>12}".format(
            name, fmt(ours), fmt(sd_gpu), fmt(sd_edge)))

    lines.append("="*70)

    # Improvement over SmartDrive GPU
    if isinstance(SMARTDRIVE_GPU.get("rmse_m"), float):
        rmse_improv = (SMARTDRIVE_GPU["rmse_m"] - metrics["rmse_m"]) \
                      / SMARTDRIVE_GPU["rmse_m"] * 100
        lines.append("\nRMSE improvement over SmartDrive GPU: {:.1f}%".format(
            rmse_improv))
        if rmse_improv > 0:
            lines.append("  -> Our system is MORE accurate")
        else:
            lines.append("  -> SmartDrive is more accurate")

    dist_improv = (metrics["mean_distance_m"]
                   - SMARTDRIVE_GPU["mean_distance_m"]) \
                  / SMARTDRIVE_GPU["mean_distance_m"] * 100
    lines.append("Distance improvement over SmartDrive GPU: {:.1f}%".format(
        dist_improv))

    summary = "\n".join(lines)
    print(summary)

    with open(summary_path, "w") as f:
        f.write(summary)
    print("\nSummary saved to {}".format(summary_path))

    return metrics


if __name__ == "__main__":
    run_evaluation()
