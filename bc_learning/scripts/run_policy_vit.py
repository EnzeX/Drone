#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import ViT_B_16_Weights
import numpy as np
import cv2
import os
from train_bc_vit import PolicyNet  

# Parameters
MODEL_PATH = os.path.expanduser("~/bc_data/airsim_bc_policy_vit.pth")
IMAGE_TOPIC = "/airsim_node/drone1/front_center/Scene"
ODOM_TOPIC = "/airsim_node/drone1/odom_local_ned"
CMD_TOPIC = "/cmd_vel"


# Initialize
bridge = CvBridge()
# model = PolicyNet()
# model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PolicyNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((120, 160)),
#     transforms.ToTensor()
# ])
weights = ViT_B_16_Weights.DEFAULT
preprocess = weights.transforms()   # includes resize + normalize


pub = None
latest_vel = None
latest_alt = None

def odom_callback(msg):
    global latest_vel, latest_alt
    v = msg.twist.twist.linear
    latest_vel = [v.x, v.y, v.z]
    latest_alt = -msg.pose.pose.position.z 
    
def image_callback(msg):
    global latest_vel, latest_alt
    max_vel = 300 
    max_alt = 1600
    if latest_vel is None or latest_alt is None:
        rospy.logwarn_throttle(5, "Waiting for odom data...")
        return
    try:
        img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        vx, vy, vz = latest_vel
        alt = latest_alt
        norm_vx = np.clip(latest_vel[0] / max_vel, -1.0, 1.0)
        norm_vy = np.clip(latest_vel[1] / max_vel, -1.0, 1.0)
        norm_vz = np.clip(latest_vel[2] / max_vel, -1.0, 1.0)
        norm_alt = np.clip(alt / max_alt, 0.0, 1.0)
        
        # img_tensor = transform(img).unsqueeze(0)  # shape: (1, 3, 120, 160)
        pil_img = transforms.ToPILImage()(img)
        img_tensor = preprocess(pil_img).unsqueeze(0)  # (1, 3, 224, 224) with normalization

        state_tensor = torch.tensor([[norm_vx, norm_vy, norm_vz, norm_alt]], dtype=torch.float32) #shape: (1, 4)

        img_tensor = img_tensor.to(device)
        state_tensor = state_tensor.to(device)

        with torch.no_grad():
            action = model(img_tensor, state_tensor).squeeze().cpu().numpy()
        
        GRAVITY_COMPENSATION = -0.25 
        action[2] += GRAVITY_COMPENSATION
        twist = Twist()
        twist.linear.x = float(action[0])
        twist.linear.y = float(action[1])
        twist.linear.z = float(action[2])
        twist.angular.z = float(action[3])
        pub.publish(twist)
        
        # Visualization
        debug_img = img.copy()
        text1 = f"VEL: [{vx:.2f}, {vy:.2f}, {vz:.2f}]  ALT: {alt:.2f}"
        text2 = f"ACTION: {[round(a,2) for a in action]}"
        cv2.putText(debug_img, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(debug_img, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        cv2.imshow("BC Policy", debug_img)
        cv2.waitKey(1)
    except Exception as e:
        rospy.logwarn(f"Prediction failed: {e}")


if __name__ == "__main__":
    rospy.init_node("airsim_bc_policy")
    pub = rospy.Publisher(CMD_TOPIC, Twist, queue_size=1)
    
    rospy.Subscriber(IMAGE_TOPIC, Image, image_callback)
    rospy.Subscriber(ODOM_TOPIC, Odometry, odom_callback)
    
    rospy.loginfo("Behavior Cloning policy is running...")
    rospy.on_shutdown(cv2.destroyAllWindows)
    rospy.spin()
    
