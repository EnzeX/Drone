#!/usr/bin/env python3
"""
ring_pid_expert.py  (v3)
FIX 1 – backwards flight after ring 9:
    Old code used current_idx % N which wraps to ring 0 (now behind you).
    New: LOOP=True flies the course repeatedly from the front.
         LOOP=False hoveres in place after the last ring.

FIX 2 – yaw instability with X+Z motion:
    Removed the yaw PID entirely. Instead we extract the drone's current
    yaw from the odometry quaternion and rotate world-frame errors into
    body frame (forward/right). This means vx always means "go the
    direction you're facing" and vy means "strafe". A gentle proportional
    yaw nudge (with deadzone) handles residual lateral drift — no integral,
    no derivative, no oscillation.

FIX 3 – only works with ring_spawner:
    On startup the expert waits 2 s for /ring_poses. If none arrives it
    falls back to querying AirSim directly for Ring_1..Ring_10, sorts them
    by NED X (front to back), and uses those as static waypoints. The same
    script now works for both scenes with zero changes.
"""

import os, math, pickle, time
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, PoseArray
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

try:
    import airsim
    AIRSIM_AVAILABLE = True
except ImportError:
    AIRSIM_AVAILABLE = False

# ── Settings ────────────────────────────────────────────────────
LOOP             = False   # True = repeat forever;  False = hover after last ring
PASS_RADIUS      = 1.5    # metres 3D

PID_X = (0.55, 0.00, 0.10)
PID_Y = (0.55, 0.00, 0.10)
PID_Z = (0.65, 0.01, 0.12)
MAX_VX, MAX_VY, MAX_VZ = 1.5, 1.0, 1.0

YAW_GAIN     = 0.4   # body-frame lateral error → angular.z
YAW_MAX      = 0.5
YAW_DEADZONE = 0.3   # metres — ignore lateral error smaller than this

GRAVITY_FF       = -0.25
MIN_SPEED_RECORD =  0.05

RING_NAME_PREFIX = "Ring_"
NUM_RINGS        = 10
POSES_TIMEOUT    = 2.0   # seconds to wait for /ring_poses before static fallback

# ── PID ─────────────────────────────────────────────────────────
class PID:
    def __init__(self, kp, ki, kd, lo, hi):
        self.kp,self.ki,self.kd,self.lo,self.hi = kp,ki,kd,lo,hi
        self.I = self.prev = 0.0; self.t = None
    def reset(self): self.I = self.prev = 0.0; self.t = None
    def step(self, e):
        now = time.time()
        dt  = max((now - self.t) if self.t else 0.05, 1e-4); self.t = now
        self.I += e * dt
        d = (e - self.prev) / dt; self.prev = e
        return float(np.clip(self.kp*e + self.ki*self.I + self.kd*d, self.lo, self.hi))

# ── Global state ─────────────────────────────────────────────────
bridge = CvBridge(); data = []
latest_image = latest_vel = latest_alt = drone_pos = None
drone_yaw    = 0.0
ring_poses_live = []   # raw from /ring_poses (ENU z)
waypoints       = []   # (x, y, z) NED — updated each frame if live
using_live      = False
current_idx     = 0    # only ever increments
course_done     = False

pid_x = PID(*PID_X, -MAX_VX, MAX_VX)
pid_y = PID(*PID_Y, -MAX_VY, MAX_VY)
pid_z = PID(*PID_Z, -MAX_VZ, MAX_VZ)

# ── Callbacks ────────────────────────────────────────────────────
def quat_to_yaw(ox, oy, oz, ow):
    return math.atan2(2*(ow*oz + ox*oy), 1 - 2*(oy*oy + oz*oz))

def image_callback(msg):
    global latest_image
    try: latest_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    except: pass

def odom_callback(msg):
    global latest_vel, latest_alt, drone_pos, drone_yaw
    v = msg.twist.twist.linear; p = msg.pose.pose.position; q = msg.pose.pose.orientation
    latest_vel = [v.x, v.y, v.z]; latest_alt = -p.z; drone_pos = [p.x, p.y, p.z]
    drone_yaw  = quat_to_yaw(q.x, q.y, q.z, q.w)

def ring_poses_callback(msg):
    global ring_poses_live
    ring_poses_live = [(p.position.x, p.position.y, p.position.z) for p in msg.poses]

# ── Static fallback ──────────────────────────────────────────────
def query_static_rings():
    if not AIRSIM_AVAILABLE:
        rospy.logerr("airsim module not available for static fallback."); return []
    try:
        c = airsim.MultirotorClient(); c.confirmConnection()
        all_obj = c.simListSceneObjects()
        names = [f"{RING_NAME_PREFIX}{i}" for i in range(1, NUM_RINGS+1)]
        names = [n for n in names if n in all_obj]
        if len(names) < 2:
            names = sorted([n for n in all_obj if RING_NAME_PREFIX in n])[:NUM_RINGS]
        if not names:
            rospy.logerr(f"No rings with prefix '{RING_NAME_PREFIX}' found."); return []
        poses = []
        for n in names:
            p = c.simGetObjectPose(n).position
            poses.append((p.x_val, p.y_val, p.z_val))
        poses.sort(key=lambda p: p[0])   # sort front-to-back by NED X
        rospy.loginfo(f"Static fallback: {len(poses)} rings loaded from AirSim")
        for i,(x,y,z) in enumerate(poses):
            rospy.loginfo(f"  [{i}] ({x:.1f}, {y:.1f}, {z:.1f}) NED")
        return poses
    except Exception as e:
        rospy.logerr(f"Static ring query failed: {e}"); return []

# ── Waypoint logic ───────────────────────────────────────────────
def get_target():
    """Return current target (x,y,z) NED. Refreshes live waypoints each call."""
    global waypoints
    if using_live and ring_poses_live:
        waypoints = [(x, y, -z) for x, y, z in ring_poses_live]  # ENU→NED z
    if not waypoints:
        return None
    n = len(waypoints)
    idx = current_idx % n if LOOP else min(current_idx, n - 1)
    return waypoints[idx]

def check_advance():
    global current_idx, course_done
    if course_done: return
    target = get_target()
    if target is None or drone_pos is None: return
    dist = math.sqrt(sum((a-b)**2 for a,b in zip(target, drone_pos)))
    if dist < PASS_RADIUS:
        current_idx += 1
        n = len(waypoints)
        if current_idx >= n and not LOOP:
            course_done = True
            rospy.loginfo("All rings passed. Hovering. Ctrl-C to save."); return
        rospy.loginfo(f"Passed ring! Next: #{current_idx % max(n,1)}  total={current_idx}")
        pid_x.reset(); pid_y.reset(); pid_z.reset()

# ── Control (FIX 2) ──────────────────────────────────────────────
def compute_action(target):
    tx, ty, tz = target
    # World-frame errors
    ex_w = tx - drone_pos[0]
    ey_w = ty - drone_pos[1]
    ez   = tz - drone_pos[2]
    # Rotate into drone body frame using current yaw
    cy, sy = math.cos(drone_yaw), math.sin(drone_yaw)
    ex_b =  ex_w * cy + ey_w * sy   # body forward
    ey_b = -ex_w * sy + ey_w * cy   # body right
    vx = pid_x.step(ex_b)
    vy = pid_y.step(ey_b)
    vz = float(np.clip(pid_z.step(ez) + GRAVITY_FF, -MAX_VZ, MAX_VZ))
    # Gentle yaw: proportional only, with deadzone — avoids oscillation
    wz = float(np.clip(YAW_GAIN * ey_b, -YAW_MAX, YAW_MAX)) if abs(ey_b) > YAW_DEADZONE else 0.0
    return [vx, vy, vz, wz]

# ── Debug overlay ────────────────────────────────────────────────
def draw_debug(img, action, target):
    out = img.copy(); h, w = out.shape[:2]
    n = max(len(waypoints), 1)
    mode = "LIVE" if using_live else "STATIC"
    lbl  = current_idx % n if LOOP else min(current_idx, n-1)
    cv2.putText(out, f"[{mode}] Ring#{lbl}  passed={current_idx}  samples={len(data)}",
                (10,26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,200,255), 2)
    if latest_vel:
        vx,vy,vz = latest_vel
        cv2.putText(out, f"VEL:[{vx:.2f},{vy:.2f},{vz:.2f}] ALT:{latest_alt:.2f} YAW:{math.degrees(drone_yaw):.0f}d",
                    (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 2)
    cv2.putText(out, f"CMD vx={action[0]:.2f} vy={action[1]:.2f} vz={action[2]:.2f} wz={action[3]:.2f}",
                (10,74), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,0), 2)
    if target and drone_pos:
        tx,ty,tz = target
        dist = math.sqrt((tx-drone_pos[0])**2+(ty-drone_pos[1])**2+(tz-drone_pos[2])**2)
        cv2.putText(out, f"ERR x={tx-drone_pos[0]:.1f} y={ty-drone_pos[1]:.1f} z={tz-drone_pos[2]:.1f} dist={dist:.1f}m",
                    (10,98), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,255,0), 2)
    if course_done:
        cv2.putText(out, "COURSE DONE — HOVERING",
                    (w//2-120, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    cx,cy2 = w//2, h-40
    cv2.arrowedLine(out,(cx,cy2),(int(cx+action[1]*30),int(cy2-action[0]*30)),(0,255,128),2,tipLength=0.35)
    cv2.circle(out,(cx,cy2),4,(255,255,255),-1)
    return out

# ── Main ─────────────────────────────────────────────────────────
def main():
    global using_live, waypoints
    rospy.init_node("ring_pid_expert", anonymous=False)
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    rospy.Subscriber("/airsim_node/drone1/front_center/Scene", Image,     image_callback)
    rospy.Subscriber("/airsim_node/drone1/odom_local_ned",     Odometry,  odom_callback)
    rospy.Subscriber("/ring_poses",                            PoseArray, ring_poses_callback)
    rate = rospy.Rate(20)

    rospy.loginfo("Waiting for image + odom...")
    while not rospy.is_shutdown():
        if latest_image is not None and latest_vel is not None: break
        rate.sleep()

    rospy.loginfo(f"Checking for /ring_poses for {POSES_TIMEOUT}s...")
    t0 = time.time()
    while not rospy.is_shutdown():
        if ring_poses_live:
            using_live = True
            rospy.loginfo(f"LIVE mode: {len(ring_poses_live)} moving rings"); break
        if time.time() - t0 > POSES_TIMEOUT:
            rospy.logwarn("No /ring_poses — STATIC fallback via AirSim")
            waypoints = query_static_rings(); break
        rate.sleep()

    if not using_live and not waypoints:
        rospy.logerr("No waypoints from either source. Exiting."); return
    rospy.loginfo(f"Ready. LOOP={LOOP}.")

    try:
        while not rospy.is_shutdown():
            if course_done:
                pub.publish(Twist())
                if latest_image is not None:
                    cv2.imshow("Ring PID", draw_debug(latest_image,[0,0,0,0],None))
                    cv2.waitKey(1)
                rate.sleep(); continue
            check_advance()
            target = get_target()
            if target is None:
                rospy.logwarn_throttle(2.0,"No target"); rate.sleep(); continue
            action = compute_action(target)
            twist = Twist()
            twist.linear.x=action[0]; twist.linear.y=action[1]
            twist.linear.z=action[2]; twist.angular.z=action[3]
            pub.publish(twist)
            spd = math.hypot(latest_vel[0],latest_vel[1]) if latest_vel else 0
            if latest_image is not None and spd >= MIN_SPEED_RECORD:
                data.append((latest_image.copy(),list(action),list(latest_vel),latest_alt))
            if latest_image is not None:
                cv2.imshow("Ring PID", draw_debug(latest_image,action,target))
                cv2.waitKey(1)
            rate.sleep()
    finally:
        path = os.path.expanduser("~/bc_data/airsim_ring_expert_data.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path,"wb") as f: pickle.dump(data,f)
        rospy.loginfo(f"Saved {len(data)} samples -> {path}")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()