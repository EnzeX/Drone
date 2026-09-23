#!/usr/bin/env python3
"""
pointcloud_eval.py
==================
Eval-only: depth -> point cloud reconstruction + true surface coverage metrics.

Why this exists: viewpoint spheres are only a proxy metric. They do not tell you
how much of the tree surface was actually observed, whether the orbit radius was
reasonable, or whether the drone kept re-observing the same side. This module
reconstructs what the camera truly saw and quantifies it.

This is not used in the training loop because it is too slow, and accumulated
coverage is non-Markovian.

Metrics:
  azimuth_coverage_pct  Fraction of each tree's azimuth that was observed
                        (36 bins of 10 degrees) - true coverage
  redundancy            Average number of points per covered azimuth bin
                        High redundancy + low coverage = repeated viewing
                        of the same side
  mean_orbit_radius_m   Mean orbit radius while near the tree
                        Quantifies "orbiting too far to reach viewpoints"
"""

import math
import os

import airsim
import cv2
import numpy as np

from orchard_env import VIEWPOINT_MIN_DIST, VIEWPOINT_MAX_DIST

# ── Tunables ────────────────────────────────────────────────────────────────
VOXEL_SIZE        = 0.15   # m - voxel downsampling to control memory usage
PIXEL_STRIDE      = 2      # sample one pixel every N pixels to reduce cost
MAX_RANGE_M       = 12.0   # discard points beyond this range; far depth is noisy
MIN_RANGE_M       = 0.3
GROUND_CLEARANCE  = 0.5    # world NED z < -0.5 counts as tree; filters ground
MAX_HEIGHT_M      = 8.0    # z > -8 filters sky and outliers
TREE_ASSIGN_RADIUS = 2.5   # m - point must be within this horizontal distance
                            # of a tree center to count as that tree's surface
N_AZIMUTH_BINS    = 36     # one bin per 10 degrees
N_AZ_BINS         = 36     # 与 azimuth coverage 保持一致
REVISIT_GAP_STEPS = 15     # 同一 bin 两次观测间隔超过此步数 → 算两次独立访问
N_HEIGHT_BINS     = 8      # 高度分箱数;与方位一起构成 36×8=288 格展开网格
HEIGHT_MIN_M      = 0.5    # 与 GROUND_CLEARANCE 一致
HEIGHT_MAX_M      = 5.0    # 实测树高:基准 b≈4.8,× scale(0.85–1.15)→ 4.1–5.5
N_CELLS           = N_AZIMUTH_BINS * N_HEIGHT_BINS
COV_SNAPSHOT_EVERY = 10    # 每 N 步记录一次覆盖率-预算曲线采样点
BUDGET_TARGETS    = (40.0, 55.0, 70.0)
ENGAGE_RADIUS     = 6.5    # m - radius considered "engaged with this tree"
                            # for orbit-radius statistics
INCIDENCE_MAX_DEG = 60.0   # 视线与表面法线夹角超此值=掠射,不算有效观测。
                            # 让"悬停一侧"无法刷满覆盖,必须真绕。
COVERAGE_MAX_VIEW_DIST = 7.0  # 相机离树超过这个距离,不计入覆盖。
                               # 12m 外扫一眼不算有效巡检。


def _quat_to_R(q):
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


class PointCloudAccumulator:
    """Incrementally accumulate back-projected depth points and compute coverage
    metrics at the end of an episode."""

    def __init__(self, env, camera_name="front_center"):
        self.env = env
        self.camera_name = camera_name
        self._intrinsics = None      # (fx, fy, cx, cy) for the RESIZED depth image
        self.reset()

    # ---------------------------------------------------------------- #
    def reset(self):
        self.voxels = set()          # quantized voxel coordinates
        self.points = []             # retained world-frame points after sampling
        self.colors = []             # RGB per point (matches self.points)
        self.traj = []               # drone trajectory
        self.surf_bins = {}          # {tree_idx: set(点方位扇区)} —— 入射角过滤后
        self.view_bins = {}          # {tree_idx: set(相机站位方位扇区)}
        self.tree_pts  = {}          # {tree_idx: 有效观测点数}
        self._radius_samples = {}    # {tree_idx: [dist,...]}
        self._step_idx      = 0
        self._bin_obs_steps = {}     # {tree_idx: {bin_k: [step, ...]}}
        self.surf_cells = {}         # {tree_idx: set((az_bin, h_bin))} 展开网格覆盖
        self.cell_hits  = {}         # {tree_idx: {(az,h): 点数}} 供展开图着色
        self.cov_curve  = []         # [(path_m, step, az_cov_pct, cell_cov_pct)]
        self._path_len  = 0.0
        self._last_xy   = None

    # ---------------------------------------------------------------- #
    def _ensure_intrinsics(self):
        """Infer resized-image fx/fy from native resolution and camera HFOV.
        Note: when the native image is resized to (IMG_W, IMG_H), the aspect
        ratio may change, so fx and fy must be scaled independently rather than
        assuming fx == fy."""
        if self._intrinsics is not None:
            return self._intrinsics

        from orchard_env import IMG_H, IMG_W

        resp = self.env.client.simGetImages([
            airsim.ImageRequest(self.camera_name, airsim.ImageType.DepthPlanar, True, False)
        ])[0]
        w_native, h_native = resp.width, resp.height

        info = self.env.client.simGetCameraInfo(self.camera_name)
        hfov = math.radians(info.fov)

        fx_native = (w_native / 2.0) / math.tan(hfov / 2.0)
        fy_native = fx_native                      # assume square pixels natively
        fx = fx_native * (IMG_W / float(w_native))
        fy = fy_native * (IMG_H / float(h_native))
        cx, cy = IMG_W / 2.0, IMG_H / 2.0

        self._intrinsics = (fx, fy, cx, cy)
        print(f"[PCD] intrinsics: fx={fx:.1f} fy={fy:.1f} "
              f"(native {w_native}x{h_native}, hfov={info.fov:.1f}deg)")
        return self._intrinsics

    # ---------------------------------------------------------------- #
    def update(self):
        """Call once after each env.step()."""
        from orchard_env import DEPTH_MAX_M

        depth_n = getattr(self.env, "_last_depth_n", None)
        if depth_n is None:
            return

        fx, fy, cx, cy = self._ensure_intrinsics()

        cam = self.env.client.simGetCameraInfo(self.camera_name)
        cam_pos = np.array([cam.pose.position.x_val,
                            cam.pose.position.y_val,
                            cam.pose.position.z_val], dtype=np.float64)
        R = _quat_to_R(cam.pose.orientation)

        state = self.env.client.getMultirotorState()
        p = state.kinematics_estimated.position
        self.traj.append((p.x_val, p.y_val, p.z_val))

        if self._last_xy is not None:
            self._path_len += math.hypot(p.x_val - self._last_xy[0],
                                         p.y_val - self._last_xy[1])
        self._last_xy = (p.x_val, p.y_val)

        drone_x, drone_y = p.x_val, p.y_val
        ori = state.kinematics_estimated.orientation
        _, _, yaw = airsim.to_eularian_angles(ori)

        d = depth_n[::PIXEL_STRIDE, ::PIXEL_STRIDE] * DEPTH_MAX_M
        h, w = d.shape
        vs, us = np.mgrid[0:h, 0:w]
        us = us * PIXEL_STRIDE
        vs = vs * PIXEL_STRIDE

        valid = (d > MIN_RANGE_M) & (d < MAX_RANGE_M) & (d < DEPTH_MAX_M * 0.999)
        if not valid.any():
            self._accumulate_radius(p.x_val, p.y_val)
            return

        z_c = d[valid]                                   # forward (DepthPlanar is orthogonal distance)
        x_c = (us[valid] - cx) / fx * z_c                # right
        y_c = (vs[valid] - cy) / fy * z_c                # down
        pts_cam = np.stack([z_c, x_c, y_c], axis=1)      # AirSim body/cam: [forward, right, down]

        rng = np.sqrt(z_c**2 + x_c**2 + y_c**2)         # 真实射线距离(欧氏)

        pts_w = pts_cam @ R.T + cam_pos

        # Filter ground, outliers, and far points by true ray distance.
        keep = (pts_w[:, 2] < -GROUND_CLEARANCE) & (pts_w[:, 2] > -MAX_HEIGHT_M) & (rng < MAX_RANGE_M)
        pts_w = pts_w[keep]
        if len(pts_w) == 0:
            self._accumulate_radius(p.x_val, p.y_val)
            return

        # ── 颜色(供 CloudCompare 查看) ──────────────────────────────────────
        from orchard_env import IMG_H, IMG_W
        raw = getattr(self.env, "_last_raw_img", None)
        cols = None
        if raw is not None:
            small = cv2.cvtColor(cv2.resize(raw, (IMG_W, IMG_H)), cv2.COLOR_BGR2RGB)
            cols = small[::PIXEL_STRIDE, ::PIXEL_STRIDE][valid][keep]

        # ── 覆盖统计:入射角过滤 + 相机站位方位 ─────────────────────────────
        for i, (tx, ty) in enumerate(self.env.tree_positions_xy):
            if math.hypot(cam_pos[0] - tx, cam_pos[1] - ty) > COVERAGE_MAX_VIEW_DIST:
                continue
            ddx = pts_w[:, 0] - tx
            ddy = pts_w[:, 1] - ty
            dh  = np.hypot(ddx, ddy)
            m   = dh < TREE_ASSIGN_RADIUS
            if not m.any():
                continue
            # 水平法线 ≈ 树心指向该点;视线 = 该点指向相机
            nx = ddx[m] / np.maximum(dh[m], 1e-6)
            ny = ddy[m] / np.maximum(dh[m], 1e-6)
            vx = cam_pos[0] - pts_w[m, 0]
            vy = cam_pos[1] - pts_w[m, 1]
            vn = np.maximum(np.hypot(vx, vy), 1e-6)
            ok = (nx * vx + ny * vy) / vn > math.cos(math.radians(INCIDENCE_MAX_DEG))
            if not ok.any():
                continue
            ang = np.arctan2(ddy[m][ok], ddx[m][ok])
            b = ((ang + math.pi) / (2 * math.pi) * N_AZIMUTH_BINS).astype(int) % N_AZIMUTH_BINS
            self.surf_bins.setdefault(i, set()).update(b.tolist())
            self.tree_pts[i] = self.tree_pts.get(i, 0) + int(ok.sum())

            # 高度分箱 → (方位, 高度) 展开网格。z 是 NED,取负得到离地高度。
            hgt = -pts_w[m][ok][:, 2]
            hb = np.clip(((hgt - HEIGHT_MIN_M) / (HEIGHT_MAX_M - HEIGHT_MIN_M)
                          * N_HEIGHT_BINS).astype(int), 0, N_HEIGHT_BINS - 1)
            cells = self.surf_cells.setdefault(i, set())
            hits  = self.cell_hits.setdefault(i, {})
            for az_k, h_k in zip(b.tolist(), hb.tolist()):
                cells.add((az_k, h_k))
                hits[(az_k, h_k)] = hits.get((az_k, h_k), 0) + 1

            cam_ang = math.atan2(cam_pos[1] - ty, cam_pos[0] - tx)
            cb = int((cam_ang + math.pi) / (2 * math.pi) * N_AZIMUTH_BINS) % N_AZIMUTH_BINS
            self.view_bins.setdefault(i, set()).add(cb)

        # ── 体素下采样(同时存颜色) ───────────────────────────────────────────
        q = np.floor(pts_w / VOXEL_SIZE).astype(np.int64)
        for j, (key, pt) in enumerate(zip(map(tuple, q), pts_w)):
            if key not in self.voxels:
                self.voxels.add(key)
                self.points.append(pt)
                self.colors.append(cols[j] if cols is not None else np.array([200, 200, 200]))

        self._accumulate_radius(p.x_val, p.y_val)

        # ── ORACLE 式有效观测记录(距离带 + FOV):用于 revisit 统计 ─────────
        for ti, (tx, ty) in enumerate(self.env.tree_positions_xy):
            d_tree = math.hypot(drone_x - tx, drone_y - ty)
            if not (VIEWPOINT_MIN_DIST <= d_tree <= VIEWPOINT_MAX_DIST):
                continue
            facing, _ = self.env._is_facing_tree(drone_x, drone_y, yaw, ti)
            if not facing:
                continue
            ang = math.atan2(drone_y - ty, drone_x - tx)
            k = int((ang + math.pi) / (2 * math.pi) * N_AZ_BINS) % N_AZ_BINS
            self._bin_obs_steps.setdefault(ti, {}).setdefault(k, []).append(self._step_idx)

        if self._step_idx % COV_SNAPSHOT_EVERY == 0:
            self.cov_curve.append((
                round(self._path_len, 2),
                self._step_idx,
                self._snapshot_cov(self.surf_bins,  N_AZIMUTH_BINS),
                self._snapshot_cov(self.surf_cells, N_CELLS),
            ))
        self._step_idx += 1

    def _accumulate_radius(self, dx, dy):
        for i, (tx, ty) in enumerate(self.env.tree_positions_xy):
            dist = math.hypot(dx - tx, dy - ty)
            if dist < ENGAGE_RADIUS:
                self._radius_samples.setdefault(i, []).append(dist)

    def _snapshot_cov(self, store, denom):
        """当前所有树的平均覆盖率(%)。"""
        n = len(self.env.tree_positions_xy)
        if n == 0:
            return 0.0
        tot = sum(len(store.get(i, set())) for i in range(n))
        return round(100.0 * tot / (n * denom), 2)

    def _budget_to(self, target_pct, use_cells=True):
        """达到 target_pct 平均覆盖率所需的飞行路径长度(m);未达到返回 None。"""
        col = 3 if use_cells else 2
        for row in self.cov_curve:
            if row[col] >= target_pct:
                return row[0]
        return None

    def _revisit_stats(self, tree_idx):
        """把每个方位 bin 的观测步序列按时间间隔切成独立访问段。
        revisit_ratio: 1.0 = 单圈无重复;2.0 = 整棵树绕了两圈。"""
        bins = self._bin_obs_steps.get(tree_idx, {})
        if not bins:
            return {"revisit_ratio": 0.0, "revisited_bins_pct": 0.0,
                    "visit_segments": 0, "observed_bins": 0}
        total_segments = revisited_bins = 0
        for steps in bins.values():
            segs = 1 + sum(1 for a, b in zip(steps, steps[1:])
                           if b - a > REVISIT_GAP_STEPS)
            total_segments += segs
            if segs > 1:
                revisited_bins += 1
        n = len(bins)
        return {
            "revisit_ratio":      round(total_segments / n, 2),
            "revisited_bins_pct": round(100.0 * revisited_bins / n, 1),
            "visit_segments":     total_segments,
            "observed_bins":      n,
        }

    # ---------------------------------------------------------------- #
    def compute_metrics(self):
        """Assign accumulated point cloud points to trees and compute azimuth
        coverage metrics."""
        trees  = self.env.tree_positions_xy
        bins   = {i: self.surf_bins.get(i, set())  for i in range(len(trees))}
        views  = {i: self.view_bins.get(i, set())  for i in range(len(trees))}
        counts = {i: self.tree_pts.get(i, 0)       for i in range(len(trees))}
        cells  = {i: self.surf_cells.get(i, set()) for i in range(len(trees))}

        per_tree = []
        for i in range(len(trees)):
            n_bins = len(bins[i])
            cov = 100.0 * n_bins / N_AZIMUTH_BINS
            radii = self._radius_samples.get(i, [])
            per_tree.append({
                "tree_idx": i,
                "azimuth_coverage_pct": round(cov, 1),
                "cell_coverage_pct": round(100.0 * len(cells[i]) / N_CELLS, 2),
                "view_dir_coverage_pct": round(100.0 * len(views[i]) / N_AZIMUTH_BINS, 1),
                "surface_points": counts[i],
                "point_density_per_bin": round(counts[i] / N_AZ_BINS, 1),
                **self._revisit_stats(i),
                "mean_orbit_radius_m": round(float(np.mean(radii)), 2) if radii else None,
                "min_orbit_radius_m": round(float(np.min(radii)), 2) if radii else None,
            })

        traj = np.array(self.traj) if self.traj else np.zeros((0, 3))
        path_len = float(np.sum(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1))) \
            if len(traj) > 1 else 0.0
        covered = [t["azimuth_coverage_pct"] for t in per_tree]
        revisit_ratios = [t["revisit_ratio"] for t in per_tree if t["observed_bins"] > 0]

        self._per_tree = per_tree
        metrics = {
            "mean_azimuth_coverage_pct": round(float(np.mean(covered)), 1) if covered else 0.0,
            "min_azimuth_coverage_pct": round(float(np.min(covered)), 1) if covered else 0.0,
            "mean_revisit_ratio": round(float(np.mean(revisit_ratios)), 2) if revisit_ratios else 0.0,
            "trees_never_visited": sum(1 for t in per_tree if t["observed_bins"] == 0),
            "mean_cell_coverage_pct": round(float(np.mean(
                [t["cell_coverage_pct"] for t in per_tree])), 2) if per_tree else 0.0,
            "min_cell_coverage_pct": round(float(np.min(
                [t["cell_coverage_pct"] for t in per_tree])), 2) if per_tree else 0.0,
            "coverage_curve": self.cov_curve,
            "total_surface_points": int(sum(counts.values())),
            "cloud_points": len(self.points),
            "path_length_m": round(path_len, 1),
            "per_tree": per_tree,
        }
        for tgt in BUDGET_TARGETS:
            metrics[f"path_to_{int(tgt)}_pct_m"] = self._budget_to(tgt)
        self._last_metrics = metrics
        return metrics

    # ---------------------------------------------------------------- #
    def save_ply(self, path):
        pts  = np.array(self.points) if self.points else np.zeros((0, 3))
        cols = np.array(self.colors, dtype=np.uint8) if self.colors else np.zeros((0, 3), np.uint8)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(pts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for (x, y, z), c in zip(pts, cols):
                f.write(f"{x:.3f} {y:.3f} {z:.3f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

    def save_plot(self, path, title="", info=None):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[PCD] matplotlib is not installed; skipping plot generation "
                  "(pip install matplotlib)")
            return

        pts = np.array(self.points) if self.points else np.zeros((0, 3))
        traj = np.array(self.traj) if self.traj else np.zeros((0, 3))

        fig = plt.figure(figsize=(16, 9))
        gs  = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1],
                               hspace=0.30, wspace=0.22)
        ax1 = fig.add_subplot(gs[0, 0])   # 俯视 X-Y
        ax2 = fig.add_subplot(gs[0, 1])   # 覆盖率 vs 路径预算
        ax3 = fig.add_subplot(gs[1, :])   # 侧视 X-Z

        # ── ax1: 俯视点云 ──────────────────────────────────────────────────
        if len(pts):
            sc = ax1.scatter(pts[:, 0], pts[:, 1], c=-pts[:, 2], s=1.5,
                             cmap="viridis", alpha=0.6)
            plt.colorbar(sc, ax=ax1, label="height above ground (m)")
        if len(traj):
            ax1.plot(traj[:, 0], traj[:, 1], "r-", lw=1.0, alpha=0.8, label="drone path")
            ax1.plot(traj[0, 0], traj[0, 1], "go", ms=8, label="start")

        for i, (tx, ty) in enumerate(self.env.tree_positions_xy):
            ax1.add_patch(plt.Circle((tx, ty), TREE_ASSIGN_RADIUS,
                                     fill=False, ls="--", ec="gray", lw=0.8))
            ax1.plot(tx, ty, "k^", ms=9)
            ax1.annotate(f"T{i}", (tx, ty), textcoords="offset points",
                         xytext=(6, 6), fontsize=9)

        ax1.set_xlabel("X (m, NED forward)")
        ax1.set_ylabel("Y (m, NED right)")
        ax1.set_title(f"Reconstructed point cloud (top-down){title}")
        ax1.axis("equal")
        ax1.legend(loc="best", fontsize=8)
        ax1.grid(alpha=0.3)

        # ── ax2: 覆盖率 vs 飞行预算 ──────────────────────────────────────
        if self.cov_curve:
            cc = np.array([(r[0], r[2], r[3]) for r in self.cov_curve])
            ax2.plot(cc[:, 0], cc[:, 2], lw=1.6, label="cell (36×8)")
            ax2.plot(cc[:, 0], cc[:, 1], lw=1.2, ls="--", alpha=0.7,
                     label="azimuth only (36, legacy)")
            for tgt in BUDGET_TARGETS:
                ax2.axhline(tgt, ls=":", c="#888888", lw=0.8)
                b = self._budget_to(tgt)
                if b is not None:
                    ax2.annotate(f"{tgt:g}% @ {b:.0f}m", (b, tgt), fontsize=7,
                                 xytext=(3, 3), textcoords="offset points")
            ax2.set_xlabel("path length flown (m)")
            ax2.set_ylabel("mean coverage (%)")
            ax2.set_title("Coverage vs. flight budget")
            ax2.set_ylim(0, 100)
            ax2.legend(fontsize=7, loc="lower right")
            ax2.grid(alpha=0.3)

        # ── ax3: 侧视图 ───────────────────────────────────────────────────
        if len(pts):
            ax3.scatter(pts[:, 0], -pts[:, 2], s=1.0, c="tab:green", alpha=0.35)
        if len(traj):
            ax3.plot(traj[:, 0], -traj[:, 2], "r-", lw=1.0, alpha=0.9)
        ax3.axhspan(HEIGHT_MIN_M, HEIGHT_MAX_M, color="tab:blue", alpha=0.06)
        ax3.set_xlabel("X (m, NED forward)")
        ax3.set_ylabel("height above ground (m)")
        ax3.set_title("Side view — vertical coverage band (fixed-altitude flight)")
        ax3.grid(alpha=0.3)

        if info:
            m = self._last_metrics
            budget_text = "  ".join(
                f"{tgt:g}%@{m[f'path_to_{int(tgt)}_pct_m']}m"
                for tgt in BUDGET_TARGETS
            )
            lines = [
                f"outcome   {info.get('done_reason', '-')}",
                f"reward    {info.get('total_reward', '-'):>8}     steps  {info.get('steps', '-')}",
                f"viewpoints{info.get('views_covered', '-')}/{info.get('total_views', '-')}"
                f"  ({info.get('coverage_pct', '-')}%)",
                f"az cov    mean {m['mean_azimuth_coverage_pct']}%   min {m['min_azimuth_coverage_pct']}%",
                f"cell cov  mean {m['mean_cell_coverage_pct']}%  min {m['min_cell_coverage_pct']}%",
                f"budget    {budget_text}",
                f"path      {m['path_length_m']} m    revisit {m.get('mean_revisit_ratio', '-')}x",
                f"min tree dist {info.get('min_tree_dist_m', '-')} m"
                f"    smooth {info.get('action_smoothness', '-')} (raw)",
            ]
            fig.subplots_adjust(bottom=0.16)
            fig.text(0.012, 0.012, "\n".join(lines), family="monospace",
                     fontsize=9, va="bottom", ha="left",
                     bbox=dict(boxstyle="round,pad=0.5", fc="#f7f7f7", ec="#aaaaaa"))

        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=130)
        plt.close(fig)

    def save_unrolled(self, path, title=""):
        """每棵树的展开柱面覆盖图:横轴方位角,纵轴高度,颜色=观测点密度。
        空洞一眼可见,并且与 cell_coverage_pct 完全自洽。"""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        n = len(self.env.tree_positions_xy)
        if n == 0:
            return
        ncol = min(4, n)
        nrow = int(math.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.6 * nrow),
                                 squeeze=False)

        for i in range(nrow * ncol):
            ax = axes[i // ncol][i % ncol]
            if i >= n:
                ax.axis("off")
                continue
            grid = np.zeros((N_HEIGHT_BINS, N_AZIMUTH_BINS))
            for (az_k, h_k), c in self.cell_hits.get(i, {}).items():
                grid[h_k, az_k] = c
            cov = 100.0 * np.count_nonzero(grid) / N_CELLS
            ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                      extent=[0, 360, HEIGHT_MIN_M, HEIGHT_MAX_M])
            ax.set_title(f"T{i}  cell cov {cov:.1f}%", fontsize=9)
            ax.set_xticks([0, 90, 180, 270, 360])
            if i % ncol == 0:
                ax.set_ylabel("height (m)", fontsize=8)
            if i // ncol == nrow - 1:
                ax.set_xlabel("azimuth (deg)", fontsize=8)
            ax.tick_params(labelsize=7)

        fig.suptitle(f"Unrolled per-tree surface coverage{title}  "
                     f"(dark = never observed)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=130)
        plt.close(fig)
