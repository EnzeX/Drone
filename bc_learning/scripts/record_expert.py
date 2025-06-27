#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import pickle
import time
import os

bridge = CvBridge()
data = []

latest_image = None
latest_vel = None
latest_alt = None

def image_callback(msg):
    global latest_image
    try:
        latest_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    except:
        rospy.logwarn("Failed to convert image")

def cmd_callback(msg):
    global latest_image, latest_vel, latest_alt
    if latest_image is not None and latest_vel is not None and latest_alt is not None:
        action = [msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z]
        data.append((latest_image.copy(), action, latest_vel.copy(), latest_alt))
        rospy.loginfo(f"Saved action: {action}, vel: {latest_vel}, alt: {latest_alt}, total: {len(data)}")
        # 显示图像 + 状态信息
        vx, vy, vz = [0 if abs(v) < 0.01 else round(v, 2) for v in latest_vel]
        debug_img = latest_image.copy()
        text = f"VEL: [{vx:.2f}, {vy:.2f}, {vz:.2f}]  ALT: {latest_alt:.2f}"
        cv2.putText(debug_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Recording Expert", debug_img)
        cv2.waitKey(1)

def odom_callback(msg):
    global latest_vel, latest_alt
    v = msg.twist.twist.linear
    latest_vel = [v.x, v.y, v.z]
    latest_alt = -msg.pose.pose.position.z  # AirSim Z 是负的，Tello 用正高度

if __name__ == "__main__":
    rospy.init_node("expert_data_recorder", anonymous=True)

    rospy.Subscriber("/airsim_node/drone1/front_center/Scene", Image, image_callback)
    rospy.Subscriber("/cmd_vel", Twist, cmd_callback)
    rospy.Subscriber("/airsim_node/drone1/odom_local_ned", Odometry, odom_callback)
    
    rospy.loginfo("Recording expert data... Ctrl+C to stop.")
    rate = rospy.Rate(10)  # 防止 CPU 空转

    try:
        while not rospy.is_shutdown():
            rate.sleep()
    finally:
        save_dir = os.path.expanduser("~/bc_data")
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"airsim_expert_data.pkl")
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        rospy.loginfo(f"Saved {len(data)} samples to {filename}")
        cv2.destroyAllWindows()

