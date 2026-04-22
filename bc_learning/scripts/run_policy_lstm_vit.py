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

from train_bc_lstm_vit import LSTMViTPolicyNet

# Parameters
MODEL_PATH = os.path.expanduser("~/bc_data/airsim_bc_policy_lstm_vit.pth")
IMAGE_TOPIC = "/airsim_node/drone1/front_center/Scene"
ODOM_TOPIC = "/airsim_node/drone1/odom_local_ned"
CMD_TOPIC = "/cmd_vel"

# Initialize
bridge = CvBridge()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
checkpoint = torch.load(MODEL_PATH, map_location=device)
model = LSTMViTPolicyNet(
    hidden_size=checkpoint['hidden_size'],
    num_lstm_layers=checkpoint['num_lstm_layers']
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ViT preprocessing
weights = ViT_B_16_Weights.DEFAULT
preprocess = weights.transforms()

# Global state
pub = None
latest_vel = None
latest_alt = None
hidden_state = None


def reset_memory():
    """Reset LSTM memory"""
    global hidden_state
    hidden_state = model.init_hidden(batch_size=1, device=device)
    rospy.loginfo("🧠 Memory reset")


def odom_callback(msg):
    global latest_vel, latest_alt
    v = msg.twist.twist.linear
    latest_vel = [v.x, v.y, v.z]
    latest_alt = -msg.pose.pose.position.z


def image_callback(msg):
    global latest_vel, latest_alt, hidden_state
    
    max_vel = 300 
    max_alt = 1600
    
    if latest_vel is None or latest_alt is None:
        rospy.logwarn_throttle(5, "Waiting for odom data...")
        return
    
    if hidden_state is None:
        reset_memory()
    
    try:
        # Convert image
        img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        
        # Get state
        vx, vy, vz = latest_vel
        alt = latest_alt
        norm_vx = np.clip(vx / max_vel, -1.0, 1.0)
        norm_vy = np.clip(vy / max_vel, -1.0, 1.0)
        norm_vz = np.clip(vz / max_vel, -1.0, 1.0)
        norm_alt = np.clip(alt / max_alt, 0.0, 1.0)
        
        # Preprocess with ViT transforms
        pil_img = transforms.ToPILImage()(img)
        img_tensor = preprocess(pil_img).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 3, 224, 224]
        
        state_tensor = torch.tensor([[norm_vx, norm_vy, norm_vz, norm_alt]], 
                                     dtype=torch.float32).unsqueeze(0).to(device)  # [1, 1, 4]
        
        # Forward pass with memory
        with torch.no_grad():
            action_output, hidden_state = model(img_tensor, state_tensor, hidden_state)
            action = action_output.squeeze(0).squeeze(0).cpu().numpy()
        
        # Gravity compensation
        GRAVITY_COMPENSATION = -0.25 
        action[2] += GRAVITY_COMPENSATION
        
        # Publish
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
        text3 = f"MEMORY: ViT+LSTM (h_norm={torch.norm(hidden_state[0]).item():.2f})"
        
        cv2.putText(debug_img, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(debug_img, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        cv2.putText(debug_img, text3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        cv2.imshow("BC ViT+LSTM Policy", debug_img)
        key = cv2.waitKey(1)
        
        if key == ord('r'):
            reset_memory()
            
    except Exception as e:
        rospy.logwarn(f"Prediction failed: {e}")


if __name__ == "__main__":
    rospy.init_node("airsim_bc_lstm_vit_policy")
    pub = rospy.Publisher(CMD_TOPIC, Twist, queue_size=1)
    
    rospy.Subscriber(IMAGE_TOPIC, Image, image_callback)
    rospy.Subscriber(ODOM_TOPIC, Odometry, odom_callback)
    
    rospy.loginfo("🚁 ViT+LSTM Behavior Cloning policy running...")
    rospy.loginfo("📝 Memory-augmented with pretrained vision features")
    rospy.loginfo("⌨️  Press 'r' to reset memory")
    
    rospy.on_shutdown(cv2.destroyAllWindows)
    rospy.spin()
