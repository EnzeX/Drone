#!/usr/bin/env python3
"""
orchard_env.py
==============
Gymnasium environment for the AirSim orchard inspection task.

Built directly on the working ball-collection version.
Tree coverage viewpoints provide the primary learning signal.
Apple gaze code is commented out.

Reward summary:
  PRIMARY:
    +1.0  per tree coverage viewpoint reached while facing the tree
    +2.0  tree completion bonus
    +small yaw shaping while orbiting the current target tree
    −0.3  one-shot trunk zone entry penalty

"""

import math
import random
import time

import airsim
import cv2
import gymnasium as gym
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  TUNABLE PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
CRUISE_ALT  = 2.5   # target flight altitude (m above spawn)
CLOCK_SPEED = 5.0

# ORACLE 式三条件判定:距离带 + FOV + 未访问扇区
VIEWPOINT_MIN_DIST = 3.0    # 太近看不全整棵树
VIEWPOINT_MAX_DIST = 5.5    # 太远分辨率不够
BASE_TRUNK_RADIUS = 0.3
DRONE_RADIUS_MARGIN = 0.25
WARNING_EXTRA = 1.3
DANGER_EXTRA = 0.5
# Coverage viewpoint parameters
VIEWPOINT_RADIUS = 4.0
TARGET_ENGAGE_RADIUS = 6.5 #VIEWPOINT_MAX_DIST + 2.0   # ~5m: locked on current tree while inside, prevents mid-orbit target switch
TARGET_SWITCH_MARGIN = 2.0                       # new target must be this many metres closer before switching, eliminates transit jitter
VIEWPOINT_COUNT = 6
VIEWPOINT_REACH_DIST = 0.8
VIEWPOINT_REWARD = 1.0
VIEWPOINT_FACING_DEG = 45.0

TRUNK_WARNING_WEIGHT = 0.02
TRUNK_DANGER_WEIGHT = 0.08
DISTANCE_PENALTY = 0.5
COLLISION_PENALTY = 3.0
ANG_RATE_PENALTY = 0.02

TREE_DONE_THRESHOLD = VIEWPOINT_COUNT
TREE_DONE_REWARD = 2.0
PROGRESS_REWARD_WEIGHT = 0.0 #05

# Orchard row randomization
ORCHARD_NUM_ROWS = 2
ORCHARD_X_START = 6.0
ORCHARD_X_SPACING = 8.0
ORCHARD_ROW_Y = [-4.0, 4.0]
ORCHARD_X_JITTER = 0.8
ORCHARD_Y_JITTER = 0.6
ORCHARD_MIN_TREE_DIST = 5.0
ORCHARD_YAW_MIN_DEG = -180.0
ORCHARD_YAW_MAX_DEG = 180.0
ORCHARD_SCALE_MIN = 0.85
ORCHARD_SCALE_MAX = 1.15
LAYOUT_MODE = "grid"          # "grid" = two-row orchard; "scatter" = free scatter
SCATTER_N_MIN, SCATTER_N_MAX = 3, 6
SCATTER_X_RANGE = (5.0, 32.0)
SCATTER_Y_RANGE = (-9.0, 9.0)

RESET_SLEEP_SCALE = 1.0 / CLOCK_SPEED

# Apple gaze parameters (disabled)
# ENABLE_APPLE_REWARD = False
# APPLE_XY_OFFSET   = 1.0    # metres FBLR from trunk centre
# APPLE_Z_OFFSET    = -2.5   # metres NED (negative = up) from tree actor origin
# GAZE_HALF_ANGLE   = 30.0   # degrees — forward cone half-angle
# INSPECT_MIN_DIST  = 0.5    # metres horizontal — too close
# INSPECT_MAX_DIST  = 2.0    # metres horizontal — too far
# INSPECT_STEPS     = 3      # consecutive steps in cone → apple collected
# APPLE_GAZE_REWARD = 2.0    # reward per apple gazed at (less than a ball)

IMG_C, IMG_W, IMG_H = 4, 80, 64   # RGB(3) + depth(1)
N_IMG = IMG_C * IMG_W * IMG_H
DEPTH_MAX_M = 20.0                 # clip/normalize depth to [0,1] (near→0, far→1)
CAMERA_HFOV_DEG      = 90.0                  # must match front_center FOV in settings.json
DEPTH_BLOB_NEAR_N    = 12.0 / DEPTH_MAX_M   # only blobs closer than ~12 m count as trees
DEPTH_BLOB_MIN_AREA  = 30                    # px; smaller blobs are noise at 64×80
DEPTH_BLOB_MATCH_DEG = 35.0                  # blob bearing vs GT bearing tolerance (deg)
STATE_DIM = 9
OBS_DIM = N_IMG + STATE_DIM

MAX_FORWARD_SPEED = 1.2
MAX_SIDE_SPEED = 0.8
MAX_YAW_RATE_DEG = 25.0
MAX_YAW_RATE_RAD = math.radians(MAX_YAW_RATE_DEG)
ALT_KP = 2.5
MAX_Z_SPEED = 2.0
CONTROL_DT = 0.15            # velocity command duration (simulated seconds)

TRAINING_MODE = True
DEBUG_MARKERS = False
VERBOSE_ENV_DEBUG = False
# If SHOW_AIRSIM_MARKERS=True, red/yellow/green AirSim debug markers may be visible
# in the policy camera image. Use it only for debugging. For real training, keep it False.
SHOW_AIRSIM_MARKERS = DEBUG_MARKERS and not TRAINING_MODE
SHOW_DEBUG_OVERLAY = True
ENABLE_TREE_RANDOMIZATION = True

STAGNATION_TIMEOUT_S = 200.0  # simulated seconds without a new viewpoint -> truncate


# ─────────────────────────────────────────────────────────────────────────────
#  SCENE HELPERS  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def _get_tree_positions(client):
    """Returns [(ue_object_name, x, y, z), ...] sorted by x. Keeping the UE
    actor name alongside the position (rather than a separate lookup) is what
    lets OrchardEnv.tree_object_names stay aligned with tree_positions_xy —
    both come from this same sorted list, so index i always refers to the
    same physical tree in both."""
    trees = []
    names = []
    for pattern in ("Tree.*", ".*[Tt]ree.*"):
        names = client.simListSceneObjects(pattern)
        if names:
            break
    for name in names:
        try:
            pose = client.simGetObjectPose(name)
            if pose.position.x_val < -100.0:
                continue
            trees.append((name,
                          pose.position.x_val,
                          pose.position.y_val,
                          pose.position.z_val))
        except Exception:
            pass
    trees.sort(key=lambda t: t[1])
    return trees


def _generate_coverage_viewpoints(tree_positions_xy):
    """每棵树周向切 VIEWPOINT_COUNT 个扇区。判定不再是"撞到某个点",
    而是 ORACLE 式三条件:距离带内 + 树心在 FOV 内 + 该扇区未访问。
    bx/by 只留作可视化标记。"""
    viewpoints = []
    angle_offset = math.pi / VIEWPOINT_COUNT

    for i, (tx, ty) in enumerate(tree_positions_xy):
        for k in range(VIEWPOINT_COUNT):
            angle = 2 * math.pi * k / VIEWPOINT_COUNT + angle_offset
            viewpoints.append({
                "bx": tx + VIEWPOINT_RADIUS * math.cos(angle),   # 仅可视化
                "by": ty + VIEWPOINT_RADIUS * math.sin(angle),   # 仅可视化
                "tree_idx": i,
                "view_idx": k,
                "collected": False,
                "reward": VIEWPOINT_REWARD,
                "label": f"tree{i}_sec{k}",
            })

    return viewpoints


# ─────────────────────────────────────────────────────────────────────────────
#  APPLE HELPERS  (disabled)
# ─────────────────────────────────────────────────────────────────────────────
# def _apple_positions(tree_x, tree_y, tree_z):
#     """4 apple world-NED positions: front, back, right, left."""
#     d, dz = APPLE_XY_OFFSET, APPLE_Z_OFFSET
#     return [
#         (tree_x + d, tree_y,     tree_z + dz),   # front
#         (tree_x - d, tree_y,     tree_z + dz),   # back
#         (tree_x,     tree_y + d, tree_z + dz),   # right
#         (tree_x,     tree_y - d, tree_z + dz),   # left
#     ]


# def _quat_to_forward_horizontal(q):
#     """Drone forward direction projected onto horizontal plane (2-D unit vector)."""
#     w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
#     fx   = 1 - 2*(y*y + z*z)
#     fy   = 2*(x*y + w*z)
#     norm = math.sqrt(fx*fx + fy*fy) + 1e-9
#     return fx/norm, fy/norm


# def _in_inspection_cone(drone_pos, fwd_x, fwd_y, apple_pos):
#     """
#     Horizontal-only cone check.
#     Returns (in_cone: bool, horiz_dist: float)
#     """
#     dx         = apple_pos[0] - drone_pos[0]
#     dy         = apple_pos[1] - drone_pos[1]
#     horiz_dist = math.sqrt(dx*dx + dy*dy)
#     if horiz_dist < 1e-6:
#         return False, 0.0
#     dot        = fwd_x*(dx/horiz_dist) + fwd_y*(dy/horiz_dist)
#     cos_thresh = math.cos(math.radians(GAZE_HALF_ANGLE))
#     in_cone    = dot > cos_thresh and INSPECT_MIN_DIST <= horiz_dist <= INSPECT_MAX_DIST
#     return in_cone, horiz_dist


def _depth_blob_bearings(depth_n):
    """从归一化深度图(HxW,近→0 远→1)提取候选树 blob。
    返回 [(bearing_rad, dist_m, area_px), ...];bearing +右/-左,相对相机朝向(=body frame)。
    纯几何,零学习——就是真机 RGB-D 感知模块会输出的东西。"""
    h, w = depth_n.shape
    mask = (depth_n < DEPTH_BLOB_NEAR_N).astype(np.uint8)
    if mask.sum() == 0:
        return []
    n_lbl, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    half_fov = math.radians(CAMERA_HFOV_DEG) * 0.5
    blobs = []
    for lbl in range(1, n_lbl):                        # 0 = background
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < DEPTH_BLOB_MIN_AREA:
            continue
        cx = float(centroids[lbl][0])                  # centroid column
        bearing = (cx - w / 2.0) / (w / 2.0) * half_fov
        dist = float(np.median(depth_n[labels == lbl])) * DEPTH_MAX_M
        blobs.append((bearing, dist, area))
    return blobs


# ─────────────────────────────────────────────────────────────────────────────
#  ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────
class OrchardEnv(gym.Env):
    """
    Observation:
      [RGB image (3x64x80 flattened)] +
      [vx_body_n, vy_body_n, vz_n, alt_n,
       dx_body_n, dy_body_n, sin_yaw, cos_yaw, yaw_rate_n]

    Action:
      [forward_body_velocity, side_body_velocity, yaw_rate] in [-1, 1]
    """

    metadata = {}

    def __init__(self):
        super().__init__()

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        self.spawn_z           = None
        self.cruise_z          = None
        self.spawn_xy          = None
        self.tree_positions_xy = []
        self.tree_object_names = []
        self.max_x_bound       = float('inf')
        self.min_x_bound       = float('-inf')
        self.max_y_bound       = float('inf')
        self.min_y_bound       = float('-inf')
        self.step_count        = 0
        self.viewpoints             = []
        self._stagnation_steps = 0
        self._sim_t_last = None
        self._sim_t_last_progress = None
        self._step_dt_ema = None
        self._last_collected   = 0
        self._trees_in_penalty = set()
        self._last_done_reason = None
        self._last_raw_img     = None
        self._last_depth_n     = None
        self._last_state_info  = {}
        self._printed_marker_debug = False
        self.target_tree_idx   = 0
        self._prev_x           = None
        self._yaw_reward_by_tree = {}
        self._rewarded_done_trees = set()
        self._prev_cmd_action = np.zeros(3, dtype=np.float32)
        self.tree_scales       = []

        # AirSim's has_collided flag is sticky (stays True until overwritten by
        # a newer collision event) and is not reliably cleared by client.reset().
        # We track the timestamp of the last collision we've already accounted
        # for (or the stale one seen right after takeoff) and only treat a
        # collision as "new" if its time_stamp is more recent than this.
        self._collision_ts_baseline = 0

        # Apple gaze state (disabled)
        # self.apple_trees       = []   # [{'trunk':..., 'apples':[...]}]
        # self.apple_collected   = []   # [tree_idx][apple_idx] bool
        # self.gaze_streak       = []   # [tree_idx][apple_idx] int

        # Vector dropout curriculum: probability of zeroing dx/dy in obs
        self.vector_dropout_p = 0.0

    # ------------------------------------------------------------------ #
    #  TREE RANDOMIZATION
    # ------------------------------------------------------------------ #
    def _print_tree_positions(self, tag):
        if not VERBOSE_ENV_DEBUG:
            return
        trees = _get_tree_positions(self.client)
        print(f"[TREES {tag}] "
              f"{[(name, round(x, 2), round(y, 2), round(z, 2)) for name, x, y, z in trees]}")

    def _debug_scene_tree_names(self):
        if not VERBOSE_ENV_DEBUG:
            return
        for pattern in ("Tree.*", ".*[Tt]ree.*"):
            names = self.client.simListSceneObjects(pattern)
            print(f"[TREE NAMES] pattern={pattern} count={len(names)}")
            for name in names[:20]:
                pose = self.client.simGetObjectPose(name)
                print(
                    f"  {name}: "
                    f"({pose.position.x_val:.2f}, "
                    f"{pose.position.y_val:.2f}, "
                    f"{pose.position.z_val:.2f})"
                )

    def _tree_collected_count(self, tree_idx):
        prefix = f"tree{tree_idx}_sec"
        return sum(
            1 for b in self.viewpoints
            if b["label"].startswith(prefix) and b["collected"]
        )

    def _is_tree_done(self, tree_idx):
        return self._tree_collected_count(tree_idx) >= TREE_DONE_THRESHOLD

    def _update_target_tree(self, drone_x, drone_y):
        """Target = 最近的未完成树(贪心)。黏滞:当前目标在 TARGET_ENGAGE_RADIUS 内且没绕完
        就保持,不让目标绕一半乱切;否则重选最近未完成树。返回当前目标"刚绕完"的 done 奖励。
        无固定 0->1->2 顺序——和 depth-blob"找最近树"的行为一致。"""
        reward_bonus = 0.0
        cur = self.target_tree_idx

        # 当前目标刚绕完 → 发一次 done 奖励
        if (cur < len(self.tree_positions_xy)
                and self._is_tree_done(cur)
                and cur not in self._rewarded_done_trees):
            self._rewarded_done_trees.add(cur)
            reward_bonus += TREE_DONE_REWARD
            print(f"[TREE DONE] tree{cur} "
                  f"{self._tree_collected_count(cur)}/{VIEWPOINT_COUNT}")

        # 黏滞:当前目标没绕完且还贴着它,保持不切
        if (cur < len(self.tree_positions_xy)
                and not self._is_tree_done(cur)):
            tx, ty = self.tree_positions_xy[cur]
            if math.hypot(drone_x - tx, drone_y - ty) < TARGET_ENGAGE_RADIUS:
                return reward_bonus

        # 否则:重选最近的未完成树(带滞回,避免 transit 段来回跳)
        best_idx, best_d = None, float("inf")
        for i, (tx, ty) in enumerate(self.tree_positions_xy):
            if self._is_tree_done(i):
                continue
            d = math.hypot(drone_x - tx, drone_y - ty)
            if d < best_d:
                best_d, best_idx = d, i

        # 当前目标还没绕完时,只有新目标近过它一定 margin 才切换;否则保持当前,消除抖动
        if (cur < len(self.tree_positions_xy) and not self._is_tree_done(cur)
                and best_idx is not None and best_idx != cur):
            cx, cy = self.tree_positions_xy[cur]
            cur_d = math.hypot(drone_x - cx, drone_y - cy)
            if best_d > cur_d - TARGET_SWITCH_MARGIN:
                best_idx = cur   # 差距不够大,保持当前目标

        if best_idx is not None and best_idx != self.target_tree_idx:
            self.target_tree_idx = best_idx
            print(f"[TARGET] Now targeting nearest tree {best_idx} (d={best_d:.1f} m)")

        return reward_bonus

    @property
    def all_trees_done(self):
        n = len(self.tree_positions_xy)
        return n > 0 and all(self._is_tree_done(i) for i in range(n))

    def _randomize_trees(self):
        names = []
        for pattern in ("Tree.*", ".*[Tt]ree.*"):
            names = self.client.simListSceneObjects(pattern)
            if names:
                break

        if not names:
            print("[ENV] No tree objects found for randomization.")
            return

        # Sort by original x so trees remain in orchard order
        tree_infos = []
        for name in names:
            pose = self.client.simGetObjectPose(name)
            tree_infos.append((name, pose.position.x_val, pose.position.y_val, pose.position.z_val))

        tree_infos.sort(key=lambda t: (t[1], t[2]))

        if LAYOUT_MODE == "scatter":
            placed = []
            active_entries = []
            parked_x = -200.0
            n_active = min(len(tree_infos), random.randint(SCATTER_N_MIN, SCATTER_N_MAX))
            active_indices = set(random.sample(range(len(tree_infos)), n_active))

            for idx, (name, old_x, old_y, old_z) in enumerate(tree_infos):
                if idx not in active_indices:
                    hidden_pose = airsim.Pose(
                        airsim.Vector3r(parked_x - 5.0 * idx, old_y, old_z),
                        airsim.to_quaternion(0, 0, 0),
                    )
                    self.client.simSetObjectPose(name, hidden_pose, teleport=True)
                    self.client.simSetObjectScale(
                        name,
                        airsim.Vector3r(1.0, 1.0, 1.0),
                    )
                    continue

                placed_ok = False
                for _ in range(50):
                    new_x = random.uniform(*SCATTER_X_RANGE)
                    new_y = random.uniform(*SCATTER_Y_RANGE)
                    if all(
                        math.hypot(new_x - px, new_y - py) > ORCHARD_MIN_TREE_DIST
                        for px, py in placed
                    ):
                        placed.append((new_x, new_y))
                        yaw_deg = random.uniform(ORCHARD_YAW_MIN_DEG, ORCHARD_YAW_MAX_DEG)
                        yaw_rad = math.radians(yaw_deg)
                        scale = random.uniform(ORCHARD_SCALE_MIN, ORCHARD_SCALE_MAX)

                        pose = airsim.Pose(
                            airsim.Vector3r(new_x, new_y, old_z),
                            airsim.to_quaternion(0, 0, yaw_rad),
                        )

                        changed = self.client.simSetObjectPose(name, pose, teleport=True)
                        self.client.simSetObjectScale(
                            name,
                            airsim.Vector3r(scale, scale, scale),
                        )
                        updated_pose = self.client.simGetObjectPose(name)
                        updated_scale = self.client.simGetObjectScale(name)

                        moved = (
                            abs(updated_pose.position.x_val - new_x) < 1e-2 and
                            abs(updated_pose.position.y_val - new_y) < 1e-2
                        )
                        active_entries.append((new_x, new_y, scale))
                        placed_ok = True

                        if VERBOSE_ENV_DEBUG:
                            print(
                                f"[TREE MOVE] {name}: "
                                f"scatter old=({old_x:.2f},{old_y:.2f},{old_z:.2f}) -> "
                                f"target=({new_x:.2f},{new_y:.2f},{old_z:.2f}) "
                                f"yaw={yaw_deg:.1f}deg "
                                f"scale={scale:.2f} "
                                f"api_changed={changed} moved={moved} "
                                f"actual_scale=({updated_scale.x_val:.2f},"
                                f"{updated_scale.y_val:.2f},"
                                f"{updated_scale.z_val:.2f})"
                            )
                        break

                if not placed_ok:
                    hidden_pose = airsim.Pose(
                        airsim.Vector3r(parked_x - 5.0 * idx, old_y, old_z),
                        airsim.to_quaternion(0, 0, 0),
                    )
                    self.client.simSetObjectPose(name, hidden_pose, teleport=True)
                    self.client.simSetObjectScale(
                        name,
                        airsim.Vector3r(1.0, 1.0, 1.0),
                    )
                    print(f"[ENV] Failed to place scatter tree {name}; parked it off-scene.")

            active_entries.sort(key=lambda t: t[0])
            self.tree_scales = [scale for _, _, scale in active_entries]
            print(
                f"[ENV] Randomized scatter orchard: {len(active_entries)}/{len(names)} active trees"
            )
            self._print_tree_positions("inside scatter randomize")
            return

        placed = []
        active_entries = []
        n = len(tree_infos)
        n_rows = ORCHARD_NUM_ROWS
        n_cols = math.ceil(n / n_rows)

        for idx, (name, old_x, old_y, old_z) in enumerate(tree_infos):
            col = idx // n_rows
            row = idx % n_rows

            base_x = ORCHARD_X_START + col * ORCHARD_X_SPACING
            base_y = ORCHARD_ROW_Y[row]

            placed_ok = False
            for _ in range(50):
                new_x = base_x + random.uniform(-ORCHARD_X_JITTER, ORCHARD_X_JITTER)
                new_y = base_y + random.uniform(-ORCHARD_Y_JITTER, ORCHARD_Y_JITTER)

                if all(math.hypot(new_x - px, new_y - py) > ORCHARD_MIN_TREE_DIST
                       for px, py in placed):
                    placed.append((new_x, new_y))
                    yaw_deg = random.uniform(ORCHARD_YAW_MIN_DEG, ORCHARD_YAW_MAX_DEG)
                    yaw_rad = math.radians(yaw_deg)
                    scale = random.uniform(ORCHARD_SCALE_MIN, ORCHARD_SCALE_MAX)

                    pose = airsim.Pose(
                        airsim.Vector3r(new_x, new_y, old_z),
                        airsim.to_quaternion(0, 0, yaw_rad),
                    )

                    changed = self.client.simSetObjectPose(name, pose, teleport=True)
                    self.client.simSetObjectScale(
                        name,
                        airsim.Vector3r(scale, scale, scale),
                    )
                    updated_pose = self.client.simGetObjectPose(name)
                    updated_scale = self.client.simGetObjectScale(name)

                    moved = (
                        abs(updated_pose.position.x_val - new_x) < 1e-2 and
                        abs(updated_pose.position.y_val - new_y) < 1e-2
                    )

                    active_entries.append((new_x, new_y, scale))
                    placed_ok = True

                    if VERBOSE_ENV_DEBUG:
                        print(
                            f"[TREE MOVE] {name}: "
                            f"row={row} col={col} "
                            f"old=({old_x:.2f},{old_y:.2f},{old_z:.2f}) -> "
                            f"target=({new_x:.2f},{new_y:.2f},{old_z:.2f}) "
                            f"yaw={yaw_deg:.1f}deg "
                            f"scale={scale:.2f} "
                            f"api_changed={changed} moved={moved} "
                            f"actual_scale=({updated_scale.x_val:.2f},"
                            f"{updated_scale.y_val:.2f},"
                            f"{updated_scale.z_val:.2f})"
                        )
                    break

            if not placed_ok:
                active_entries.append((old_x, old_y, 1.0))

        active_entries.sort(key=lambda t: t[0])
        self.tree_scales = [scale for _, _, scale in active_entries]

        print(f"[ENV] Randomized orchard rows: {len(placed)}/{len(names)} trees")
        self._print_tree_positions("inside row randomize")

    # ------------------------------------------------------------------ #
    #  ALTITUDE CONTROL
    # ------------------------------------------------------------------ #
    def _altitude_control(self, current_z):
        """
        Maintain CRUISE_ALT metres above episode spawn position.
        AirSim uses NED: +Z is downward.
        """
        alt = -(current_z - self.spawn_z)
        alt_err = CRUISE_ALT - alt

        vz = float(np.clip(
            -ALT_KP * alt_err,
            -MAX_Z_SPEED,
            MAX_Z_SPEED
        ))

        return alt, alt_err, vz

    # ------------------------------------------------------------------ #
    #  RESET
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        print("========== RESET CALLED ==========")
        super().reset(seed=seed)

        self.client.reset()
        time.sleep(0.5 * RESET_SLEEP_SCALE)
        self._printed_marker_debug = False
        self._debug_scene_tree_names()
        self._print_tree_positions("after reset before randomize")

        if ENABLE_TREE_RANDOMIZATION:
            self._randomize_trees()
            time.sleep(0.3 * RESET_SLEEP_SCALE)
        else:
            print("[ENV] Tree randomization disabled.")

        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        time.sleep(0.2 * RESET_SLEEP_SCALE)

        home_pose = self.client.simGetVehiclePose()
        self.spawn_z = home_pose.position.z_val
        self.spawn_xy = (
            home_pose.position.x_val,
            home_pose.position.y_val
        )
        target_z = self.spawn_z - CRUISE_ALT

        # Normal AirSim takeoff.
        self.client.takeoffAsync().join()

        # Reach CRUISE_ALT with the same controller used during RL.
        ALT_TOLERANCE = 0.05
        MAX_ALT_INIT_STEPS = 100

        for _ in range(MAX_ALT_INIT_STEPS):
            state = self.client.getMultirotorState()
            current_z = state.kinematics_estimated.position.z_val

            alt, alt_err, vz = self._altitude_control(current_z)

            if abs(alt_err) < ALT_TOLERANCE:
                break

            self.client.moveByVelocityAsync(
                0.0,
                0.0,
                vz,
                duration=CONTROL_DT,
                yaw_mode=airsim.YawMode(
                    is_rate=True,
                    yaw_or_rate=0.0
                )
            ).join()

        # Record the actual cruising position relative to episode spawn.
        cruise_pose = self.client.simGetVehiclePose()
        self.cruise_z = cruise_pose.position.z_val
        cruise_alt = -(self.cruise_z - self.spawn_z)
        print(
            f"[ALT INIT] "
            f"spawn_z={self.spawn_z:.3f}  "
            f"target_z={target_z:.3f}  "
            f"cruise_z={self.cruise_z:.3f}  "
            f"alt={cruise_alt:.3f}m"
        )

        trees = _get_tree_positions(self.client)

        if VERBOSE_ENV_DEBUG:
            print("home/spawn_z", self.spawn_z)
            print("target_z", target_z)
            print("cruise_z", self.cruise_z)
            print("tree z samples", trees[:3])

        self.tree_positions_xy = [(tx, ty) for name, tx, ty, tz in trees]
        self.tree_object_names = [name for name, tx, ty, tz in trees]
        if not hasattr(self, "tree_scales") or len(self.tree_scales) != len(self.tree_positions_xy):
            self.tree_scales = [1.0] * len(self.tree_positions_xy)

        if self.tree_positions_xy:
            max_tree_x = max(tx for tx, ty in self.tree_positions_xy)
            min_tree_x = min(tx for tx, ty in self.tree_positions_xy)

            x_margin = VIEWPOINT_RADIUS + VIEWPOINT_REACH_DIST + 5.0
            self.max_x_bound = max_tree_x + x_margin
            self.min_x_bound = min_tree_x - x_margin

            max_tree_y = max(ty for tx, ty in self.tree_positions_xy)
            min_tree_y = min(ty for tx, ty in self.tree_positions_xy)

            y_margin = VIEWPOINT_RADIUS + VIEWPOINT_REACH_DIST + 3.0
            self.max_y_bound = max_tree_y + y_margin
            self.min_y_bound = min_tree_y - y_margin
        else:
            self.max_x_bound = float('inf')
            self.min_x_bound = float('-inf')
            self.max_y_bound = float('inf')
            self.min_y_bound = float('-inf')

        self.step_count              = 0
        self.viewpoints                   = _generate_coverage_viewpoints(self.tree_positions_xy)
        self._stagnation_steps       = 0
        self._sim_t_last             = None
        self._sim_t_last_progress    = None
        self._last_collected         = 0
        self._trees_in_penalty       = set()
        self.target_tree_idx         = 0
        self._prev_x                 = self.spawn_xy[0]
        self._yaw_reward_by_tree     = {}
        self._rewarded_done_trees    = set()
        self._last_done_reason       = None
        self._prev_cmd_action        = np.zeros(3, dtype=np.float32)

        # Apple gaze init (disabled)
        # self.apple_trees = []
        # for tx, ty, tz in trees:
        #     self.apple_trees.append({
        #         'trunk':  (tx, ty, tz),
        #         'apples': _apple_positions(tx, ty, tz),
        #     })
        # n = len(self.apple_trees)
        # self.apple_collected = [[False]*4 for _ in range(n)]
        # self.gaze_streak     = [[0]*4     for _ in range(n)]

        self._visualize_viewpoints()

        print(f"[ENV] Reset — spawn_z={self.spawn_z:.2f}  "
              f"trees={len(self.tree_positions_xy)}  "
              f"viewpoints={len(self.viewpoints)}  "
              # f"apples={n*4}  "
              f"mission=inspect_all_trees  "
              f"max_x_bound={self.max_x_bound:.1f}  "
              f"y_bounds=[{self.min_y_bound:.1f}, {self.max_y_bound:.1f}]")
        if self._step_dt_ema is not None and self._step_dt_ema > 0.0:
            print(
                f"[ENV] measured step dt ≈ {self._step_dt_ema:.3f} s sim "
                f"({1.0 / self._step_dt_ema:.1f} Hz)"
            )
        print(f"[TARGET] Starting with tree {self.target_tree_idx}")

        # Anything AirSim flagged as a collision up to this point (spawn overlap,
        # a brush during arm/takeoff) is stale — baseline it away so step() only
        # reacts to genuinely new collisions during this episode.
        self._collision_ts_baseline = self.client.simGetCollisionInfo().time_stamp

        return self._get_obs(), {}

    # ------------------------------------------------------------------ #
    #  STEP
    # ------------------------------------------------------------------ #
    def step(self, action):
        ACTION_SMOOTH_ALPHA = 0.3

        raw_action = np.asarray(action, dtype=np.float32).reshape(-1)
        raw_action = np.clip(raw_action, -1.0, 1.0)

        cmd_action = raw_action.copy()
        cmd_action[:2] = (
            ACTION_SMOOTH_ALPHA * raw_action[:2]
            + (1.0 - ACTION_SMOOTH_ALPHA) * self._prev_cmd_action[:2]
        )

        # NEW: slew-limit yaw command (raw yaw was unsmoothed → hard jumps cause jitter)
        YAW_CMD_SLEW = 0.25   # max normalized yaw cmd change per step (0.25*30 = 7.5 deg/s)
        cmd_action[2] = float(np.clip(
            raw_action[2],
            self._prev_cmd_action[2] - YAW_CMD_SLEW,
            self._prev_cmd_action[2] + YAW_CMD_SLEW,
        ))

        self._prev_cmd_action = cmd_action.copy()

        vx_body = float(cmd_action[0]) * MAX_FORWARD_SPEED
        vy_body = float(cmd_action[1]) * MAX_SIDE_SPEED
        yaw_rate = float(cmd_action[2]) * MAX_YAW_RATE_DEG

        # if self.step_count % 20 == 0:
        #     print(
        #         f"[CMD] raw={raw_action}, "
        #         f"cmd_vel=[{vx_body:.2f}, {vy_body:.2f}, {yaw_rate:.1f}], "
        #     )

        state   = self.client.getMultirotorState()
        pos     = state.kinematics_estimated.position

        alt, alt_err, vz = self._altitude_control(pos.z_val)

        ori = state.kinematics_estimated.orientation
        _, _, yaw = airsim.to_eularian_angles(ori)

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        # body forward/right -> world x/y
        vx_world = cos_yaw * vx_body - sin_yaw * vy_body
        vy_world = sin_yaw * vx_body + cos_yaw * vy_body

        self.client.moveByVelocityAsync(
            vx_world,
            vy_world,
            vz,
            duration=CONTROL_DT,
            yaw_mode=airsim.YawMode(
                is_rate=True,
                yaw_or_rate=yaw_rate
            )
        ).join()

        obs                = self._get_obs()
        reward, terminated = self._compute_reward()

        sim_t = state.timestamp * 1e-9 if getattr(state, "timestamp", 0) else None
        if sim_t is None:
            sim_t = time.time() * CLOCK_SPEED
        if self._sim_t_last is not None:
            dt = sim_t - self._sim_t_last
            if 0.0 < dt < 5.0:
                self._step_dt_ema = (
                    dt if self._step_dt_ema is None
                    else 0.98 * self._step_dt_ema + 0.02 * dt
                )
        self._sim_t_last = sim_t
        if self._sim_t_last_progress is None:
            self._sim_t_last_progress = sim_t

        self.step_count += 1
        collected_now = sum(1 for b in self.viewpoints if b["collected"])
        if collected_now > self._last_collected:
            self._last_collected      = collected_now
            self._stagnation_steps    = 0
            self._sim_t_last_progress = sim_t
        else:
            self._stagnation_steps += 1
        truncated = (sim_t - self._sim_t_last_progress) > STAGNATION_TIMEOUT_S
        if truncated:
            self._last_done_reason = "stagnation_timeout"
            print(f"[ENV] Stagnation timeout after {self.step_count} steps")

        if terminated or truncated:
            print(f"[DONE] reason={self._last_done_reason}, step={self.step_count}")

        if VERBOSE_ENV_DEBUG and self.step_count < 5:
            s = self._last_state_info
            print("body vel:", s.get("vx_body", 0.0), s.get("vy_body", 0.0),
                  "yaw:", s.get("yaw", 0.0), "yaw_rate:", s.get("yaw_rate", 0.0))
            print("target body:", s.get("dx_body", 0.0), s.get("dy_body", 0.0))

        return obs, reward, terminated, truncated, {}

    # ------------------------------------------------------------------ #
    #  OBSERVATION
    # ------------------------------------------------------------------ #
    def _get_obs(self):
        img = None
        depth = None
        for attempt in range(5):
            responses = self.client.simGetImages([
                airsim.ImageRequest("front_center", airsim.ImageType.Scene,
                                    False, False),
                airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar,
                                    True, False),   # pixels_as_float=True
            ])
            r_rgb, r_depth = responses[0], responses[1]
            ok_rgb   = r_rgb.width > 0 and r_rgb.height > 0 and len(r_rgb.image_data_uint8) > 0
            ok_depth = r_depth.width > 0 and r_depth.height > 0 and len(r_depth.image_data_float) > 0
            if ok_rgb and ok_depth:
                raw = np.frombuffer(r_rgb.image_data_uint8, dtype=np.uint8)
                img = raw.reshape(r_rgb.height, r_rgb.width, 3)
                depth = np.array(r_depth.image_data_float, dtype=np.float32)
                depth = depth.reshape(r_depth.height, r_depth.width)
                break
            time.sleep(0.05 * RESET_SLEEP_SCALE)

        if img is None:
            img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
            print("[WARN] AirSim returned empty RGB — using blank frame")
        if depth is None:
            depth = np.full((IMG_H, IMG_W), DEPTH_MAX_M, dtype=np.float32)
            print("[WARN] AirSim returned empty depth — using far plane")

        self._last_raw_img = img.copy()

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (IMG_W, IMG_H))
        img_rgb = img_rgb.astype(np.float32) / 255.0                  # HxWx3, [0,1]

        depth = np.nan_to_num(depth, nan=DEPTH_MAX_M, posinf=DEPTH_MAX_M, neginf=0.0)
        depth = cv2.resize(depth, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        depth_n = np.clip(depth / DEPTH_MAX_M, 0.0, 1.0).astype(np.float32)  # HxW
        self._last_depth_n = depth_n  # expose for debug prints

        rgbd = np.concatenate([img_rgb, depth_n[..., None]], axis=2)  # HxWx4
        img_flat = np.transpose(rgbd, (2, 0, 1)).flatten()            # CHW flatten

        state = self.client.getMultirotorState()
        v     = state.kinematics_estimated.linear_velocity
        pos   = state.kinematics_estimated.position
        ori   = state.kinematics_estimated.orientation
        _, _, yaw = airsim.to_eularian_angles(ori)
        ang_vel = state.kinematics_estimated.angular_velocity
        yaw_rate = ang_vel.z_val
        alt   = -(pos.z_val - self.spawn_z)
        drone_x, drone_y = pos.x_val, pos.y_val

        d_to_target_tree = float("inf")
        if self.target_tree_idx < len(self.tree_positions_xy):
            ttx, tty = self.tree_positions_xy[self.target_tree_idx]
            d_to_target_tree = math.hypot(drone_x - ttx, drone_y - tty)

        vx_world = v.x_val
        vy_world = v.y_val
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        vx_body = cos_yaw * vx_world + sin_yaw * vy_world
        vy_body = -sin_yaw * vx_world + cos_yaw * vy_world

        vx_body_n = float(np.clip(vx_body / MAX_FORWARD_SPEED, -1.0, 1.0))
        vy_body_n = float(np.clip(vy_body / MAX_SIDE_SPEED, -1.0, 1.0))
        vz_n = float(np.clip(v.z_val / MAX_Z_SPEED, -1.0, 1.0))
        alt_n = float(np.clip(alt / 5.0, 0.0, 1.0))
        yaw_rate_n = float(np.clip(yaw_rate / MAX_YAW_RATE_RAD, -1.0, 1.0))

        candidate_balls = []
        if self.target_tree_idx < len(self.tree_positions_xy):
            prefix = f"tree{self.target_tree_idx}_sec"
            candidate_balls = [
                b for b in self.viewpoints
                if (not b["collected"]) and b["label"].startswith(prefix)
            ]
        else:
            candidate_balls = []

        nearest_ball      = None
        nearest_ball_dist = float("inf")
        for ball in candidate_balls:
            d = math.hypot(drone_x - ball["bx"], drone_y - ball["by"])
            if d < nearest_ball_dist:
                nearest_ball_dist = d
                nearest_ball = ball

        if nearest_ball is not None:
            dx_world = nearest_ball["bx"] - drone_x
            dy_world = nearest_ball["by"] - drone_y
            dx_body = cos_yaw * dx_world + sin_yaw * dy_world
            dy_body = -sin_yaw * dx_world + cos_yaw * dy_world
            dx_body_n = float(np.clip(dx_body / 10.0, -1.0, 1.0))
            dy_body_n = float(np.clip(dy_body / 10.0, -1.0, 1.0))
        else:
            dx_body = 0.0
            dy_body = 0.0
            dx_body_n = 0.0
            dy_body_n = 0.0

        # ── Approach bearing: depth blob(真机风格)+ GT fallback ──────────────
        # dx_body_n/dy_body_n 是指向目标树最近未收集 viewpoint 的 GT 特权向量。
        # 目标树在深度图里可见时,用 blob 方位替换它;只有目标出视野(比如背对着)才退回 GT。
        # LSTM 负责平滑 blob 的逐帧抖动——正是无记忆 policy 做不到的那一点。
        gt_dx_n, gt_dy_n = dx_body_n, dy_body_n

        matched_blob = None
        if self.target_tree_idx < len(self.tree_positions_xy):
            ttx, tty = self.tree_positions_xy[self.target_tree_idx]
            tdx_w, tdy_w = ttx - drone_x, tty - drone_y
            fwd =  cos_yaw * tdx_w + sin_yaw * tdy_w
            rgt = -sin_yaw * tdx_w + cos_yaw * tdy_w
            gt_bearing = math.atan2(rgt, fwd)   # GT bearing 只用来在多个 blob 里挑目标那棵

            best_err = math.radians(DEPTH_BLOB_MATCH_DEG)
            for bearing, dist, _area in _depth_blob_bearings(depth_n):
                err = abs((bearing - gt_bearing + math.pi) % (2 * math.pi) - math.pi)
                if err < best_err:
                    best_err = err
                    matched_blob = (bearing, dist)

        if matched_blob is not None:
            bearing, dist = matched_blob
            dx_body_n = float(np.clip(dist * math.cos(bearing) / 10.0, -1.0, 1.0))
            dy_body_n = float(np.clip(dist * math.sin(bearing) / 10.0, -1.0, 1.0))
        else:
            dx_body_n, dy_body_n = gt_dx_n, gt_dy_n

        # Orbit gate: zero the approach vector near the tree → force visual orbiting
        ORBIT_GATE_RADIUS = 6.0 #VIEWPOINT_RADIUS + 2.0
        if d_to_target_tree < ORBIT_GATE_RADIUS:
            dx_body_n, dy_body_n = 0.0, 0.0

        if self.vector_dropout_p > 0.0 and random.random() < self.vector_dropout_p:
            dx_body_n, dy_body_n = 0.0, 0.0

        self._last_state_info = {
            "vx_body": vx_body,
            "vy_body": vy_body,
            "vz": v.z_val,
            "alt": alt,
            "yaw": yaw,
            "yaw_rate": yaw_rate,
            "dx_body": dx_body if nearest_ball is not None else 0.0,
            "dy_body": dy_body if nearest_ball is not None else 0.0,
            "nearest_ball_label": nearest_ball["label"] if nearest_ball is not None else None,
            "nearest_ball_dist": nearest_ball_dist if nearest_ball is not None else None,
            "target_tree_idx": self.target_tree_idx,
        }

        state_vec = np.array([
            vx_body_n,
            vy_body_n,
            vz_n,
            alt_n,
            dx_body_n,
            dy_body_n,
            math.sin(yaw),
            math.cos(yaw),
            yaw_rate_n,
        ], dtype=np.float32)

        return np.concatenate([img_flat, state_vec]).astype(np.float32)

    # ------------------------------------------------------------------ #
    #  REWARD HELPERS
    # ------------------------------------------------------------------ #
    def _is_facing_tree(self, drone_x, drone_y, yaw, tree_idx):
        tx, ty = self.tree_positions_xy[tree_idx]
        desired_yaw = math.atan2(ty - drone_y, tx - drone_x)
        yaw_err = (desired_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
        return abs(yaw_err) < math.radians(VIEWPOINT_FACING_DEG), yaw_err

    def _drone_sector(self, drone_x, drone_y, tree_idx):
        """无人机当前站位落在该树的哪个周向扇区。
        与 _generate_coverage_viewpoints 的 angle_offset 保持一致。"""
        tx, ty = self.tree_positions_xy[tree_idx]
        ang = math.atan2(drone_y - ty, drone_x - tx)
        angle_offset = math.pi / VIEWPOINT_COUNT
        k = int(math.floor((ang - angle_offset) / (2 * math.pi) * VIEWPOINT_COUNT))
        return k % VIEWPOINT_COUNT

    # ------------------------------------------------------------------ #
    #  REWARD
    # ------------------------------------------------------------------ #
    def _compute_reward(self):
        state   = self.client.getMultirotorState()
        pos     = state.kinematics_estimated.position
        ori     = state.kinematics_estimated.orientation
        drone_x = pos.x_val
        drone_y = pos.y_val
        # drone_pos = [pos.x_val, pos.y_val, pos.z_val]  # Apple gaze only.

        reward     = 0.0
        terminated = False
        self._last_done_reason = None

        # Angular rate smoothness penalty (Markovian, physical, not action-space)
        ang = state.kinematics_estimated.angular_velocity
        reward -= ANG_RATE_PENALTY * (abs(ang.x_val) + abs(ang.y_val))  # roll/pitch only; yaw needed for orbiting

        progress = drone_x - getattr(self, "_prev_x", drone_x)
        progress = float(np.clip(progress, -0.2, 0.2))
        reward += PROGRESS_REWARD_WEIGHT * progress
        self._prev_x = drone_x

        if self.target_tree_idx < len(self.tree_positions_xy):
            tx, ty = self.tree_positions_xy[self.target_tree_idx]
            target_tree_dist = math.hypot(drone_x - tx, drone_y - ty)

            if target_tree_dist < VIEWPOINT_RADIUS + 0.5 and not self._is_tree_done(self.target_tree_idx):
                vel = state.kinematics_estimated.linear_velocity
                horiz_speed = math.hypot(vel.x_val, vel.y_val)
                YAW_MIN_SPEED  = 0.3   # m/s — must be orbiting, not hovering
                YAW_REWARD_CAP = 1.0   # max cumulative yaw shaping per tree
                accum = self._yaw_reward_by_tree.get(self.target_tree_idx, 0.0)
                if horiz_speed > YAW_MIN_SPEED and accum < YAW_REWARD_CAP:
                    current_yaw = airsim.to_eularian_angles(ori)[2]
                    desired_yaw = math.atan2(ty - drone_y, tx - drone_x)
                    yaw_err = (desired_yaw - current_yaw + math.pi) % (2 * math.pi) - math.pi
                    yaw_r = 0.03 * math.cos(yaw_err)
                    reward += yaw_r
                    self._yaw_reward_by_tree[self.target_tree_idx] = accum + max(0.0, yaw_r)

        # ── ORACLE 式覆盖判定:距离带 + FOV + 未访问扇区 ──────────────
        current_yaw = airsim.to_eularian_angles(ori)[2]

        if self.target_tree_idx < len(self.tree_positions_xy):
            ti = self.target_tree_idx
            ttx, tty = self.tree_positions_xy[ti]
            tree_dist = math.hypot(drone_x - ttx, drone_y - tty)
            facing_tree, yaw_err = self._is_facing_tree(
                drone_x, drone_y, current_yaw, ti
            )

            if (VIEWPOINT_MIN_DIST <= tree_dist <= VIEWPOINT_MAX_DIST) and facing_tree:
                k = self._drone_sector(drone_x, drone_y, ti)
                for vp in self.viewpoints:
                    if (vp["tree_idx"] == ti and vp["view_idx"] == k
                            and not vp["collected"]):
                        vp["collected"] = True
                        reward += vp["reward"]
                        print(
                            f"[VIEW] {vp['label']} covered! "
                            f"dist={tree_dist:.2f}, "
                            f"yaw_err={math.degrees(yaw_err):.1f}deg "
                            f"+{vp['reward']}"
                        )
                        self._visualize_viewpoints()
                        break

        reward += self._update_target_tree(drone_x, drone_y)

        if self.all_trees_done:
            terminated = True
            self._last_done_reason = "success_all_trees_inspected"
            print("[ENV] Success: all trees inspected.")

        # ── Dynamic trunk safety penalty ───────────────────────────────
        collision = self.client.simGetCollisionInfo()
        for i, (tx, ty) in enumerate(self.tree_positions_xy):
            d = math.hypot(drone_x - tx, drone_y - ty)

            scale = self.tree_scales[i] if i < len(self.tree_scales) else 1.0

            collision_r = BASE_TRUNK_RADIUS * scale
            danger_r = collision_r + DRONE_RADIUS_MARGIN + DANGER_EXTRA
            warning_r = collision_r + DRONE_RADIUS_MARGIN + WARNING_EXTRA

            # soft warning zone: encourages wider orbit, but not too strong
            if d < warning_r:
                reward -= TRUNK_WARNING_WEIGHT * (warning_r - d)
                print(f"[WARN] trunk warning tree{i} "
                      f"dist={d:.2f} scale={scale:.2f} "
                      f"collision_r={collision_r:.2f} "
                      f"reward={-TRUNK_WARNING_WEIGHT * (warning_r - d):.2f} ")

            # stronger danger zone: close to actual collision box
            if d < danger_r:
                reward -= TRUNK_DANGER_WEIGHT * (danger_r - d)
                print(f"[DANGER] trunk danger tree{i} "
                      f"dist={d:.2f} scale={scale:.2f} "
                      f"collision_r={collision_r:.2f} "
                      f"reward={-TRUNK_DANGER_WEIGHT * (danger_r - d):.2f} ")

            # one-shot severe penalty if extremely close
            if d < collision_r + DRONE_RADIUS_MARGIN:
                if i not in self._trees_in_penalty:
                    reward -= DISTANCE_PENALTY
                    self._trees_in_penalty.add(i)
                    print(
                        f"[PENALTY] trunk danger tree{i} "
                        f"dist={d:.2f} scale={scale:.2f} "
                        f"collision_r={collision_r:.2f}"
                        f" reward={-DISTANCE_PENALTY:.2f}"
                    )
            else:
                self._trees_in_penalty.discard(i)

        # ── Hard collision termination (event-based, NOT proximity-gated) ──
        # collision.has_collided is sticky in AirSim: it can still read True from
        # a stale event (e.g. brushing something during arm/takeoff) and is not
        # reliably cleared per-step or by client.reset(). Gating this purely on
        # "is the drone currently near a tree" (the old check) meant collisions
        # with the ground or anything away from a tree were silently ignored.
        # Compare time_stamp against the per-episode baseline instead, so any
        # genuinely new collision terminates the episode regardless of what it
        # hit or where the drone currently is.
        if (
            not terminated
            and collision.has_collided
            and collision.time_stamp > self._collision_ts_baseline
        ):
            terminated = True
            self._last_done_reason = "collision"
            reward -= COLLISION_PENALTY
            self._collision_ts_baseline = collision.time_stamp

            nearest_i, nearest_d = None, float("inf")
            for i, (tx, ty) in enumerate(self.tree_positions_xy):
                d = math.hypot(drone_x - tx, drone_y - ty)
                if d < nearest_d:
                    nearest_d, nearest_i = d, i
            nearest_name = (
                self.tree_object_names[nearest_i]
                if nearest_i is not None and nearest_i < len(self.tree_object_names)
                else None
            )
            print(f"[COLLISION] object={collision.object_name!r} "
                  f"nearest_tree=tree{nearest_i}({nearest_name}) "
                  f"dist={nearest_d:.2f}m -{COLLISION_PENALTY}")

        # ── Boundary termination ───────────────────────────────
        if not terminated and drone_x > self.max_x_bound:
            terminated = True
            self._last_done_reason = "out_of_x_max"
            reward -= 2.0
        if not terminated and drone_x < self.min_x_bound:
            terminated = True
            self._last_done_reason = "out_of_x_min"
            reward -= 2.0
        if not terminated and drone_y > self.max_y_bound:
            terminated = True
            self._last_done_reason = "out_of_y_max"
            reward -= 2.0
        if not terminated and drone_y < self.min_y_bound:
            terminated = True
            self._last_done_reason = "out_of_y_min"
            reward -= 2.0

        # ── Apple gaze bonus (disabled while debugging tree progression) ──
        # if ENABLE_APPLE_REWARD and not terminated:
        #     fwd_x, fwd_y = _quat_to_forward_horizontal(ori)
        #     for ti, tree in enumerate(self.apple_trees):
        #         for ai, apple_pos in enumerate(tree['apples']):
        #             if self.apple_collected[ti][ai]:
        #                 continue
        #             in_cone, hdist = _in_inspection_cone(
        #                 drone_pos, fwd_x, fwd_y, apple_pos)
        #             if in_cone:
        #                 self.gaze_streak[ti][ai] += 1
        #                 if self.gaze_streak[ti][ai] >= INSPECT_STEPS:
        #                     self.apple_collected[ti][ai] = True
        #                     self.gaze_streak[ti][ai]     = 0
        #                     reward += APPLE_GAZE_REWARD
        #                     print(f"[APPLE] Gazed tree{ti} apple{ai}  "
        #                           f"dist={hdist:.2f}m  +{APPLE_GAZE_REWARD}")
        #             else:
        #                 self.gaze_streak[ti][ai] = max(
        #                     0, self.gaze_streak[ti][ai] - 1)

        return reward, terminated

    # ------------------------------------------------------------------ #
    #  BALL VISUALIZATION
    # ------------------------------------------------------------------ #
    def _visualize_viewpoints(self):
        if not self.viewpoints or self.spawn_z is None:
            return
        self.client.simFlushPersistentMarkers()
        if not SHOW_AIRSIM_MARKERS:
            return

        marker_z = self.cruise_z if self.cruise_z is not None else self.spawn_z - CRUISE_ALT
        if VERBOSE_ENV_DEBUG and not self._printed_marker_debug:
            trees = _get_tree_positions(self.client)
            print("spawn_z", self.spawn_z)
            print("marker_z", marker_z)
            print("tree z samples", trees[:3])
            self._printed_marker_debug = True
        tree_pts = []
        for vp in self.viewpoints:
            if vp["collected"]:
                continue
            pt = airsim.Vector3r(vp["bx"], vp["by"], marker_z)
            tree_pts.append(pt)
        if tree_pts:
            self.client.simPlotPoints(
                tree_pts,
                color_rgba=[1.0, 0.0, 0.0, 1.0],
                size=20.0, duration=-1.0, is_persistent=True,
            )


    # ------------------------------------------------------------------ #
    #  VISUALIZATION
    # ------------------------------------------------------------------ #
    def show_frame(self, action=None, reward=None, window_title="RL Policy", show_depth=False):
        if self._last_raw_img is None:
            return
        vis = self._last_raw_img.copy()
        s   = self._last_state_info
        collected = sum(1 for b in self.viewpoints if b["collected"])
        total     = len(self.viewpoints)
        # apples_collected = sum(self.apple_collected[ti][ai]
        #                        for ti in range(len(self.apple_trees))
        #                        for ai in range(4))

        if SHOW_DEBUG_OVERLAY:
            cv2.putText(vis,
                f"BODY VEL: [{s.get('vx_body',0):.2f}, {s.get('vy_body',0):.2f}, "
                f"{s.get('vz',0):.2f}]  ALT: {s.get('alt',0):.2f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.putText(vis,
                f"YAW: {s.get('yaw',0):.2f}  YAW_RATE: {s.get('yaw_rate',0):.2f}  "
                f"TARGET_BODY: [{s.get('dx_body',0):.2f}, {s.get('dy_body',0):.2f}]",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        if action is not None:
            cv2.putText(vis,
                f"ACT[fwd,side,yaw]: {[round(float(a), 2) for a in action]}",
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        if reward is not None:
            cv2.putText(vis,
                f"REW: {reward:.3f}  VIEWS: {collected}/{total}  "
                # f"APPLES: {apples_collected}/{len(self.apple_trees)*4}  "
                f"STEP: {self.step_count}  DROP: {self.vector_dropout_p:.2f}",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 2)

            if s.get("nearest_ball_label") is not None:
                cv2.putText(vis,
                    f"TARGET_VIEW: {s['nearest_ball_label']}  "
                    f"DIST: {s.get('nearest_ball_dist', 0.0):.2f}",
                    (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)

        cv2.putText(
            vis,
            f"TARGET_TREE: {self.target_tree_idx}/{len(self.tree_positions_xy)}",
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

        cv2.imshow(window_title, vis)
        if show_depth and self._last_depth_n is not None:
            depth_vis = (255.0 * (1.0 - self._last_depth_n)).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
            cv2.imshow(f"{window_title} Depth", depth_vis)
        cv2.waitKey(1)

    # ------------------------------------------------------------------ #
    def close(self):
        cv2.destroyAllWindows()
        self.client.reset()
