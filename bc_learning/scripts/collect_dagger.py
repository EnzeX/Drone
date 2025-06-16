#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import torch
from torchvision import transforms
import numpy as np
import pickle
import time
import os
from train_bc import PolicyNet
from pynput import keyboard
import cv2

bridge = CvBridge()
latest_image = None
human_override = [0.0, 0.0, 0.0, 0.0]  # linear.x, y, z, angular.z

# 键盘控制映射
def on_press(key):
    try:
        if key.char == 'i': human_override[0] = 2.0     # forward
        if key.char == ',': human_override[0] = -2.0    # backward
        if key.char == 'j': human_override[3] = 1.0     # turn left
        if key.char == 'l': human_override[3] = -1.0    # turn right
        if key.char == 't': human_override[2] = 1.0     # up
        if key.char == 'b': human_override[2] = -1.0    # down
    except:
        pass

def on_release(key):
    global human_override
    human_override = [0.0, 0.0, 0.0, 0.0]
    if key == keyboard.Key.esc:
        rospy.signal_shutdown("ESC pressed to stop.")

def image_callback(msg):
    global latest_image
    try:
        latest_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    except:
        rospy.logwarn("Image conversion failed")

def overlay_info(image, pred_action, human_action):
    text1 = f"Model: x={pred_action[0]:.2f}, y={pred_action[1]:.2f}, z={pred_action[2]:.2f}, az={pred_action[3]:.2f}"
    text2 = f"Human: x={human_action[0]:.2f}, y={human_action[1]:.2f}, z={human_action[2]:.2f}, az={human_action[3]:.2f}"
    cv2.putText(image, text1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(image, text2, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return image

def main():
    rospy.init_node("dagger_visual_collector")
    rospy.Subscriber("/airsim_node/drone1/front_center/Scene", Image, image_callback)
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

    model_path = os.path.expanduser("~/airsim_bc_data/bc_policy_1.pth")
    model = PolicyNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((120, 160)),
        transforms.ToTensor()
    ])

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    rate = rospy.Rate(5)
    data = []

    rospy.loginfo("📹 可视化 DAgger 收集中，ESC 退出。")
    while not rospy.is_shutdown():
        if latest_image is None:
            rate.sleep()
            continue

        img = latest_image.copy()
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            action = model(img_tensor).squeeze().numpy()

        # 发布模型动作
        twist = Twist()
        twist.linear.x = float(action[0])
        twist.linear.y = float(action[1])
        twist.linear.z = float(action[2])
        twist.angular.z = float(action[3])
        pub.publish(twist)

        # 显示图像窗口
        display = overlay_info(img.copy(), action, human_override)
        cv2.imshow("DAgger Visual Collector", display)
        key = cv2.waitKey(1)
        if key == 27:  # ESC 键
            rospy.signal_shutdown("ESC pressed")
            break

        # 保存图像和人类动作（即使人类未操作也记录）
        expert_action = human_override.copy()
        data.append((img, expert_action))

        rate.sleep()

    # 保存数据
    save_dir = os.path.expanduser("~/airsim_bc_data")
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"dagger_data_{int(time.time())}.pkl")
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    rospy.loginfo(f"✅ Saved {len(data)} DAgger samples to {filename}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

