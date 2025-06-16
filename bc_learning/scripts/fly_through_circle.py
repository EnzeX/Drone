#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

class GreenCircleFollower:
    def __init__(self):
        rospy.init_node("green_circle_follower", anonymous=True)
        self.bridge = CvBridge()

        # 初始化订阅和发布
        rospy.Subscriber("/airsim_node/drone1/front_center/image_raw", Image, self.image_callback)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        # 图像相关
        self.img_center = (0, 0)
        self.circle_center = None
        self.current_radius = 0

        # 控制状态
        self.mode = "searching"   # 状态机：searching -> approaching -> passing -> done
        self.pass_counter = 0     # 用于 passing 模式帧计数

        rospy.loginfo("GreenCircleFollower initialized.")
        rospy.spin()

    def image_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w, _ = img.shape
        self.img_center = (w//2, h//2)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        self.circle_center = None
        self.current_radius = 0

        if contours:
            c = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(c)

            # 合理半径范围过滤（防止误识别）
            if 20 < radius < 400:
                self.circle_center = (int(x), int(y))
                self.current_radius = radius
                cv2.circle(img, self.circle_center, int(radius), (255, 0, 0), 2)

        # 调用控制逻辑
        self.control_drone()

        # 可视化中心方框
        cv2.rectangle(img, (w//2 - 80, h//2 - 80), (w//2 + 80, h//2 + 80), (0, 255, 0), 2)
        cv2.imshow("Tracking", img)
        cv2.waitKey(1)

    def control_drone(self):
        cmd = Twist()

        if self.mode == "searching":
            if self.circle_center and self.current_radius > 50:
                self.mode = "approaching"
                rospy.loginfo("Target found. Switching to approaching mode.")

        elif self.mode == "approaching":
            if self.circle_center:
                cx, cy = self.circle_center
                dx = cx - self.img_center[0]
                dy = cy - self.img_center[1]

                # 你提供的控制逻辑 + 方向已修正
                if abs(dx) > 10:
                    cmd.angular.z = 0.003 * dx
                if abs(dy) > 5:
                    cmd.linear.z = 0.008 * dy

                if self.current_radius > 350:
                    self.mode = "passing"
                    self.pass_counter = 40
                    rospy.loginfo("Switching to passing mode.")
                elif abs(dx) < 80 and abs(dy) < 80:
                    cmd.linear.x = 2.5

        elif self.mode == "passing":
            if self.pass_counter > 0:
                cmd.linear.x = 2.0
                self.pass_counter -= 1
            else:
                if not self.circle_center or self.current_radius < 30:
                    self.mode = "done"
                    rospy.loginfo("Pass complete. Switching to done.")
        elif self.mode == "done":
            cmd = Twist()  # 停止

        self.cmd_pub.publish(cmd)

if __name__ == "__main__":
    try:
        GreenCircleFollower()
    except rospy.ROSInterruptException:
        pass

