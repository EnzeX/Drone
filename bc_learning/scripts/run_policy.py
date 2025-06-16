#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import cv2
import os
from train_bc import PolicyNet  # 复用模型定义

# 参数设置
MODEL_PATH = os.path.expanduser("~/airsim_bc_data/bc_policy.pth")
IMAGE_TOPIC = "/airsim_node/drone1/front_center/Scene"
ODOM_TOPIC = "/airsim_node/drone1/odom_local_ned"
CMD_TOPIC = "/cmd_vel"

# 初始化
bridge = CvBridge()
model = PolicyNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((120, 160)),
    transforms.ToTensor()
])

pub = None
latest_vel = None
latest_alt = None

def odom_callback(msg):
    global latest_vel, latest_alt
    v = msg.twist.twist.linear
    latest_vel = [v.x, v.y, v.z]
    latest_alt = -msg.pose.pose.position.z  # AirSim Z 为负，Tello 高度为正
def image_callback(msg):
    global latest_vel, latest_alt
    if latest_vel is None or latest_alt is None:
        rospy.logwarn_throttle(5, "等待 odom 数据...")
        return
    try:
        img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        vx, vy, vz = [0 if abs(v) < 0.01 else v for v in latest_vel]
        alt = latest_alt
        
        img_tensor = transform(img).unsqueeze(0)  # shape: (1, 3, 120, 160)
        state_tensor = torch.tensor([[vx, vy, vz, alt]], dtype=torch.float32)  # shape: (1, 4)
        with torch.no_grad():
            action = model(img_tensor, state_tensor).squeeze().numpy()
        
        GRAVITY_COMPENSATION = -0.25 
        action[2] += GRAVITY_COMPENSATION
        twist = Twist()
        twist.linear.x = float(action[0])
        twist.linear.y = float(action[1])
        twist.linear.z = float(action[2])
        twist.angular.z = float(action[3])
        pub.publish(twist)
        rospy.loginfo(f"Predicted action: {action}")
        # 可视化输出
        debug_img = img.copy()
        text1 = f"VEL: [{vx:.2f}, {vy:.2f}, {vz:.2f}]  ALT: {alt:.2f}"
        text2 = f"ACTION: {[round(a,2) for a in action]}"
        cv2.putText(debug_img, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(debug_img, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        cv2.imshow("BC Policy Debug", debug_img)
        cv2.waitKey(1)
    except Exception as e:
        rospy.logwarn(f"Prediction failed: {e}")


if __name__ == "__main__":
    rospy.init_node("bc_policy_node")
    pub = rospy.Publisher(CMD_TOPIC, Twist, queue_size=1)
    rospy.Subscriber(IMAGE_TOPIC, Image, image_callback)
    rospy.Subscriber(ODOM_TOPIC, Odometry, odom_callback)
    rospy.loginfo("Behavior Cloning policy is running...")
    rospy.on_shutdown(cv2.destroyAllWindows)
    rospy.spin()

