#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image
from djitellopy import Tello
from cv_bridge import CvBridge
import cv2

tello = Tello()
tello.connect()
tello.streamon()
print("✅ Tello connected")
print(f"🔋 battery: {tello.get_battery()}%")

def cmd_vel_callback(msg):
    lr = int(msg.linear.y * 100)
    fb = int(msg.linear.x * 100)
    ud = int(msg.linear.z * 100)
    yv = int(msg.angular.z * 100)
    tello.send_rc_control(lr, fb, ud, yv)

def event_callback(msg):
    if msg.data == "takeoff" and not tello.is_flying:
        print("🛫 take off")
        tello.takeoff()
    elif msg.data == "land" and tello.is_flying:
        print("🛬 land")
        tello.land()
        
rospy.init_node("tello_listener")

# 订阅控制话题
rospy.Subscriber("/cmd_vel", Twist, cmd_vel_callback)
rospy.Subscriber("/tello_event", String, event_callback)
# 发布速度和高度
vel_pub = rospy.Publisher("/tello/velocity", Vector3, queue_size=1)
alt_pub = rospy.Publisher("/tello/height", Float32, queue_size=1)
img_pub = rospy.Publisher("/tello/image_raw", Image, queue_size=1)
bridge = CvBridge()

rate = rospy.Rate(10)
while not rospy.is_shutdown():
    try:
        # 速度与高度发布
        vx = tello.get_speed_x() / 100.0
        vy = tello.get_speed_y() / 100.0
        vz = tello.get_speed_z() / 100.0
        h = tello.get_height() / 100.0
        vel_pub.publish(Vector3(x=vx, y=vy, z=vz))
        alt_pub.publish(h)
        
        # 图像发布
        frame = tello.get_frame_read().frame
        img_msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        img_pub.publish(img_msg)
        
    except Exception as e:
        rospy.logwarn(f"读取速度/高度失败: {e}")
    rate.sleep()
    
