#!/usr/bin/env python3
"""
run_rl.py — evaluate a trained RecurrentPPO (LSTM) policy.

Critical vs the PPO version: the LSTM hidden state must be carried across steps and
RESET at the start of every episode, else memory leaks between episodes.
"""

import os
import math
import json
from datetime import datetime

import numpy as np
import airsim
from sb3_contrib import RecurrentPPO
from orchard_env import OrchardEnv
from pointcloud_eval import BUDGET_TARGETS, PointCloudAccumulator

MODEL_PATH = os.path.expanduser("~/bc_data/rl_models/slalom_ppo_lstm_final")
SAVE_DIR   = os.path.expanduser("~/bc_data/rl_models")
N_EPISODES = 10
SHOW_EVAL_RENDER = True
SHOW_EVAL_DEPTH  = True
ENABLE_POINTCLOUD = True
PCD_DIR = os.path.expanduser("~/bc_data/rl_models/pointclouds")


def evaluate(model, env, n_episodes=10):
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset()

        pcd = PointCloudAccumulator(env) if ENABLE_POINTCLOUD else None
        if pcd:
            pcd.reset()

        # ── LSTM state: 每个 episode 开头 reset ──────────────────────────
        lstm_states   = None
        episode_start = np.ones((1,), dtype=bool)

        ep_reward, ep_steps = 0.0, 0
        min_tree_dist = float('inf')
        action_history = []

        while True:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=True,
            )
            episode_start = np.zeros((1,), dtype=bool)  # 只有第一步是 True

            obs, reward, terminated, truncated, _ = env.step(action)

            ep_reward += reward
            ep_steps  += 1
            action_history.append(np.asarray(action).copy())
            if pcd:
                pcd.update()
            if SHOW_EVAL_RENDER:
                env.show_frame(action=action, reward=reward,
                               window_title="RL Eval", show_depth=SHOW_EVAL_DEPTH)

            state   = env.client.getMultirotorState()
            pos     = state.kinematics_estimated.position
            drone_x, drone_y = pos.x_val, pos.y_val
            for tx, ty in env.tree_positions_xy:
                d = math.hypot(drone_x - tx, drone_y - ty)
                if d < min_tree_dist:
                    min_tree_dist = d

            if terminated or truncated:
                break

        collected = sum(1 for b in env.viewpoints if b["collected"])
        total     = len(env.viewpoints)
        success   = env.all_trees_done
        reason    = getattr(env, "_last_done_reason", None)
        out_of_bounds = reason in ["out_of_x_max", "out_of_x_min", "out_of_y_max", "out_of_y_min"]
        crashed   = out_of_bounds or reason == "collision"

        if len(action_history) > 1:
            diffs = [np.linalg.norm(action_history[i+1] - action_history[i])
                     for i in range(len(action_history) - 1)]
            smoothness = float(np.mean(diffs))
        else:
            smoothness = 0.0

        coverage_pct = round(100 * collected / max(1, total), 1)
        ep_info = {
            "total_reward":      round(ep_reward, 2),
            "steps":             ep_steps,
            "done_reason":       reason,
            "views_covered":     collected,
            "total_views":       total,
            "coverage_pct":      coverage_pct,
            "min_tree_dist_m":   round(min_tree_dist, 3),
            "action_smoothness": round(smoothness, 4),
        }

        pcd_metrics = None
        if pcd:
            pcd_metrics = pcd.compute_metrics()
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(PCD_DIR, exist_ok=True)
            pcd.save_plot(os.path.join(PCD_DIR, f"ep{ep+1}_{stamp}.png"),
                          title=f" — episode {ep+1}", info=ep_info)
            pcd.save_unrolled(os.path.join(PCD_DIR, f"ep{ep+1}_{stamp}_unrolled.png"),
                              title=f" — episode {ep+1}")
            pcd.save_ply(os.path.join(PCD_DIR, f"ep{ep+1}_{stamp}.ply"))
            budget_text = "  ".join(
                f"{tgt:g}%@{pcd_metrics[f'path_to_{int(tgt)}_pct_m']}m"
                for tgt in BUDGET_TARGETS
            )
            print(f"  TRUE coverage:  az={pcd_metrics['mean_azimuth_coverage_pct']}%  "
                  f"cell={pcd_metrics['mean_cell_coverage_pct']}%  "
                  f"{budget_text}  "
                  f"path={pcd_metrics['path_length_m']}m")
            for t in pcd_metrics["per_tree"]:
                print(f"    T{t['tree_idx']}: cov={t['azimuth_coverage_pct']}%  "
                      f"revisit={t['revisit_ratio']}x  r_mean={t['mean_orbit_radius_m']}")

        results.append({
            "episode":  ep + 1,
            "success":  success,
            "crashed":  crashed,
            **ep_info,
            "pointcloud": pcd_metrics,
        })

        status = "SUCCESS" if success else ("CRASH" if crashed else "TIMEOUT")
        print(f"\nEpisode {ep+1}/{n_episodes}  {status}  ({reason})")
        print(f"  Reward: {ep_reward:.1f}  Steps: {ep_steps}")
        print(f"  Coverage: {collected}/{total} ({coverage_pct}%)  MinTreeDist: {min_tree_dist:.2f} m")
        print(f"  Smoothness: {smoothness:.4f} (raw policy output)")

    return results


def summarise(results):
    n = len(results)
    success_rate = 100 * sum(r["success"] for r in results) / n
    crash_rate   = 100 * sum(r["crashed"] for r in results) / n
    avg_reward   = np.mean([r["total_reward"]      for r in results])
    avg_steps    = np.mean([r["steps"]             for r in results])
    avg_cov      = np.mean([r["coverage_pct"]      for r in results])
    avg_dist     = np.mean([r["min_tree_dist_m"]   for r in results])
    avg_smooth   = np.mean([r["action_smoothness"] for r in results])

    print("\n" + "=" * 50)
    print(f"  EVALUATION SUMMARY  ({n} episodes)")
    print("=" * 50)
    print(f"  Success rate:  {success_rate:.0f}%")
    print(f"  Crash rate:    {crash_rate:.0f}%")
    print(f"  Avg reward:    {avg_reward:.1f}")
    print(f"  Avg steps:     {avg_steps:.0f}")
    print(f"  Avg coverage:  {avg_cov:.1f}%")
    print(f"  Avg min dist:  {avg_dist:.2f} m")
    print(f"  Avg smoothness:{avg_smooth:.4f}")
    print("=" * 50)
    return {
        "n_episodes": n, "success_rate_pct": round(success_rate, 1),
        "crash_rate_pct": round(crash_rate, 1), "avg_reward": round(float(avg_reward), 2),
        "avg_steps": round(float(avg_steps), 1), "avg_coverage_pct": round(float(avg_cov), 1),
        "avg_min_tree_dist": round(float(avg_dist), 3), "avg_smoothness": round(float(avg_smooth), 4),
    }


def save_results(results, summary, save_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f"eval_{timestamp}.json")
    with open(path, "w") as f:
        json.dump({"summary": summary, "episodes": results}, f, indent=2)
    print(f"\n  Results saved to {path}")


if __name__ == "__main__":
    print(f"Loading RecurrentPPO model from {MODEL_PATH}...")
    print(f"Eval render: {'on' if SHOW_EVAL_RENDER else 'off'}")
    print(f"Eval depth : {'on' if SHOW_EVAL_DEPTH else 'off'}")
    print(f"Pointcloud : {'on' if ENABLE_POINTCLOUD else 'off'}")
    env   = OrchardEnv()
    model = RecurrentPPO.load(MODEL_PATH, env=env)
    print(f"Evaluating over {N_EPISODES} episodes...\n")
    results = evaluate(model, env, n_episodes=N_EPISODES)
    summary = summarise(results)
    save_results(results, summary, SAVE_DIR)
    env.close()
