#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import cv2
import os
from train_bc import PolicyNet  # 复用模型定义

# 参数设置
MODEL_PATH = os.path.expanduser("~/bc_data/tello_bc_policy.pth")
CMD_TOPIC = "/cmd_vel"
IMAGE_TOPIC = "/tello/image_raw"
VEL_TOPIC = "/tello/velocity"
ALT_TOPIC = "/tello/height"
ATT_TOPIC = "/tello/attitude"
ACC_TOPIC = "/tello/acceleration"

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
latest_att = None
latest_acc = None

def vel_callback(msg):
    global latest_vel
    latest_vel = [msg.x, msg.y, msg.z]

def alt_callback(msg):
    global latest_alt
    latest_alt = msg.data
    
def att_callback(msg):
    global latest_att
    latest_att = [msg.x, msg.y, msg.z]
    
def acc_callback(msg):
    global latest_acc
    latest_acc = [msg.x, msg.y, msg.z]
    
def image_callback(msg):
    global latest_vel, latest_alt, latest_att, latest_acc
    if None in (latest_vel, latest_alt, latest_att, latest_acc):
        rospy.logwarn_throttle(5, "等待 odom 数据...")
        return
    try:
        img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        vx, vy, vz = [0 if abs(v) < 0.01 else round(v, 2) for v in latest_vel]
        alt = latest_alt
        pitch, roll, yaw = latest_att
        acc_x, acc_y, acc_z = latest_acc
        
        img_tensor = transform(img).unsqueeze(0)  # shape: (1, 3, 120, 160)
        state_tensor = torch.tensor([[vx, vy, vz, alt, pitch, roll, yaw, acc_x, acc_y, acc_z]], dtype=torch.float32) 
        with torch.no_grad():
            action = model(img_tensor, state_tensor).squeeze().numpy()
            speed_scale = 2.0
            action *= speed_scale
        
#        GRAVITY_COMPENSATION = -0.01 
#        action[2] += GRAVITY_COMPENSATION
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
    rospy.init_node("tello_bc_policy")
    pub = rospy.Publisher(CMD_TOPIC, Twist, queue_size=1)
    
    rospy.Subscriber(IMAGE_TOPIC, Image, image_callback)
    rospy.Subscriber(VEL_TOPIC, Vector3, vel_callback)
    rospy.Subscriber(ALT_TOPIC, Float32, alt_callback)
    rospy.Subscriber(ATT_TOPIC, Vector3, att_callback)
    rospy.Subscriber(ACC_TOPIC, Vector3, acc_callback)
    
    rospy.loginfo("Behavior Cloning policy is running...")
    rospy.on_shutdown(cv2.destroyAllWindows)
    rospy.spin()
    
