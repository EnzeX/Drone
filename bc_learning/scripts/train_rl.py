#!/usr/bin/env python3
"""
train_rl.py — RecurrentPPO (LSTM) training for the AirSim orchard task.

Why recurrent: a memoryless policy cannot tell "orbiting the target tree" from
"orbiting a tree that's already done" (identical single-frame obs once the approach
vector is gated), nor remember which trees are finished / which way to turn to
reacquire the next tree. The LSTM hidden state carries that history.

Obs:  RGBD (4x64x80) + 9D state.   Action: [fwd_body, side_body, yaw_rate].
Approach dx/dy is privileged GT in sim, gated to 0 inside the orbit radius; in
deployment it comes from an onboard tree detector / depth blob (not GPS).
"""

import glob
import os
import re
import zipfile

import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from orchard_env import OBS_DIM, OrchardEnv
from rl_policy import ImageStateFeatureExtractor

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS / RUN CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SAVE_DIR   = os.path.expanduser("~/bc_data/rl_models")
MODEL_NAME = "slalom_ppo_lstm"          # 新血统,不会覆盖 PPO/depth 的 checkpoint


def _find_latest_checkpoint(save_dir, model_name):
    """Newest f"{model_name}_<steps>_steps.zip" in save_dir (path without the
    .zip extension), or None if none exist. Avoids hand-editing a step count
    into a path literal every time training is resumed from a newer save."""
    best_path, best_steps = None, -1
    for path in glob.glob(os.path.join(save_dir, f"{model_name}_*_steps.zip")):
        m = re.search(r"_(\d+)_steps\.zip$", path)
        if m and int(m.group(1)) > best_steps:
            best_steps, best_path = int(m.group(1)), path[:-len(".zip")]
    return best_path


# Warm-start: 把旧 PPO depth run 的视觉主干 (features_extractor) 灌进来
WARMSTART_FROM_PPO = False
PPO_CKPT_PATH = _find_latest_checkpoint(SAVE_DIR, "slalom_ppo_depth")

# Resume 一个 RecurrentPPO run(第一次不存在 → 走 fresh)
RESUME_FROM_CHECKPOINT = True
CHECKPOINT_PATH = _find_latest_checkpoint(SAVE_DIR, MODEL_NAME)
# CHECKPOINT_PATH = os.path.expanduser("~/bc_data/rl_models/slalom_ppo_lstm_1400000_steps")

USE_RENDER = False
DEBUG_STARTUP_CHECK = False
TOTAL_TIMESTEPS = 200_000

os.makedirs(SAVE_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  WARM-START  (视觉主干 only — LSTM + heads 保持随机)
# ─────────────────────────────────────────────────────────────────────────────
def warmstart_extractor_from_ppo(recurrent_model, ppo_ckpt_no_ext):
    """只把 features_extractor.* 从旧 PPO checkpoint 拷进 RecurrentPPO;LSTM 和
    policy/value head 保持随机初始化,和 LSTM 一起 co-adapt。obs 形状没变,extractor
    每个张量都 shape 对齐。"""
    if not ppo_ckpt_no_ext or not os.path.exists(ppo_ckpt_no_ext + ".zip"):
        print(f"[WARMSTART][WARN] 找不到 PPO checkpoint ({ppo_ckpt_no_ext}) — 跳过,从零训练。")
        return
    with zipfile.ZipFile(ppo_ckpt_no_ext + ".zip") as archive:
        with archive.open("policy.pth") as f:
            old_sd = torch.load(f, map_location="cpu")

    new_sd = recurrent_model.policy.state_dict()
    copied, mismatched = [], []
    for k, v_new in new_sd.items():
        if not k.startswith("features_extractor"):
            continue                       # 故意跳过 LSTM + heads
        v_old = old_sd.get(k)
        if v_old is None:
            continue
        if v_old.shape == v_new.shape:
            new_sd[k] = v_old
            copied.append(k)
        else:
            mismatched.append(k)

    recurrent_model.policy.load_state_dict(new_sd)
    print(f"[WARMSTART] features_extractor 张量拷贝: {len(copied)}")
    if mismatched:
        print(f"[WARMSTART][WARN] shape 不符,未拷贝: {mismatched}")
    if len(copied) == 0:
        print("[WARMSTART][WARN] 拷了 0 个 — key 命名对不上,别当成功!")


# ─────────────────────────────────────────────────────────────────────────────
#  CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────
class RenderCallback(BaseCallback):
    def __init__(self, render_every=200, verbose=0):
        super().__init__(verbose)
        self.render_every = render_every

    def _on_step(self) -> bool:
        if self.num_timesteps % self.render_every != 0:
            return True
        env = self.training_env.envs[0].unwrapped
        action = self.locals.get("actions")
        reward = self.locals.get("rewards")
        act = action[0] if action is not None else None
        rew = float(reward[0]) if reward is not None else None
        env.show_frame(action=act, reward=rew, window_title="RL Training")
        return True


class TaskMetricsCallback(BaseCallback):
    def __init__(self, log_every=1000, verbose=0):
        super().__init__(verbose)
        self.log_every = log_every

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_every != 0:
            return True
        env = self.training_env.envs[0].unwrapped
        views_covered = sum(1 for b in env.viewpoints if b["collected"])
        total_views   = len(env.viewpoints)
        coverage_ratio = views_covered / total_views if total_views > 0 else 0.0
        mission_complete = env.all_trees_done
        self.logger.record("task/views_covered", views_covered)
        self.logger.record("task/coverage_ratio", coverage_ratio)
        self.logger.record("task/target_tree_idx", env.target_tree_idx)
        self.logger.record("task/mission_complete", float(mission_complete))
        self.logger.record("task/stagnation_steps", env._stagnation_steps)
        return True


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("Creating environment...")
    env = OrchardEnv()

    if DEBUG_STARTUP_CHECK:
        obs, _ = env.reset()
        print("Initial obs shape:", obs.shape, " expected:", OBS_DIM)
        print("Action space:", env.action_space)
        d = env._last_depth_n
        print(
            f"[DEPTH CHECK] min={d.min():.3f} "
            f"max={d.max():.3f} mean={d.mean():.3f}"
        )

    policy_kwargs = dict(
        features_extractor_class=ImageStateFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
        lstm_hidden_size=128,
        n_lstm_layers=1,
        # shared_lstm / enable_critic_lstm 用库默认(actor 和 critic 各自一个 LSTM,推荐配置)
    )

    if (
        RESUME_FROM_CHECKPOINT and
        CHECKPOINT_PATH is not None and
        os.path.exists(CHECKPOINT_PATH + ".zip")
    ):
        print(f"[INFO] Resuming RecurrentPPO checkpoint: {CHECKPOINT_PATH}")
        model = RecurrentPPO.load(
            CHECKPOINT_PATH, env=env, device="cuda",
            tensorboard_log=os.path.join(SAVE_DIR, "tb_logs"),
        )
        model.lr_schedule = lambda _: 1e-4
        model.ent_coef = 0.007
        model.n_epochs = 3
        model.target_kl = 0.07
        reset_num_timesteps = False
    else:
        print("[INFO] Fresh RecurrentPPO run.")
        model = RecurrentPPO(
            "MlpLstmPolicy", env,
            verbose=1,
            learning_rate=1e-4,
            n_steps=512,
            batch_size=128,
            n_epochs=3,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.07,
            device="cuda",
            policy_kwargs=policy_kwargs,
            tensorboard_log=os.path.join(SAVE_DIR, "tb_logs"),
        )
        if WARMSTART_FROM_PPO:
            warmstart_extractor_from_ppo(model, PPO_CKPT_PATH)
        reset_num_timesteps = True

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000, save_path=SAVE_DIR, name_prefix=MODEL_NAME, verbose=1,
    )
    task_metrics_cb = TaskMetricsCallback(log_every=1000)
    callbacks = [checkpoint_cb, task_metrics_cb]
    if USE_RENDER:
        callbacks.append(RenderCallback(render_every=200))
    print(f"[INFO] Training render: {'on' if USE_RENDER else 'off'}")

    print("=" * 55)
    print("  Starting RecurrentPPO training (LSTM)")
    print(f"  TensorBoard: tensorboard --logdir {os.path.join(SAVE_DIR, 'tb_logs')}")
    print("  Ctrl+C saves and exits cleanly")
    print("=" * 55)

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=reset_num_timesteps,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted.")

    final_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_final")
    model.save(final_path)
    print(f"[INFO] Saved to {final_path}.zip")
    env.close()


if __name__ == "__main__":
    main()
