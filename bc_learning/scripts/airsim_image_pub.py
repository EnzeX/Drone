#!/usr/bin/env python3
import rospy
import airsim
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class AirSimImagePublisher:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        self.pub = rospy.Publisher('/airsim_node/drone1/front_center/image_raw', Image, queue_size=1)
        self.bridge = CvBridge()
        self.rate = rospy.Rate(20)

    def run(self):
        while not rospy.is_shutdown():
            responses = self.client.simGetImages([
                airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
            ])
            if responses and responses[0].height > 0:
                img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)

                ros_img = self.bridge.cv2_to_imgmsg(img_rgb, encoding="rgb8")
                self.pub.publish(ros_img)
            self.rate.sleep()

if __name__ == '__main__':
    rospy.init_node('airsim_image_pub', anonymous=True)
    node = AirSimImagePublisher()
    node.run()

