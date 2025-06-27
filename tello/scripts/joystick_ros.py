#!/usr/bin/env python3
import rospy
import pygame
import time
from geometry_msgs.msg import Twist
from std_msgs.msg import String

# 初始化 pygame 和手柄
pygame.init()
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    raise RuntimeError("❌ 未检测到任何手柄，请检查连接")

joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"🎮 检测到手柄：{joystick.get_name()}")

# 初始化 ROS
rospy.init_node("joystick_control")
pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
event_pub = rospy.Publisher("/tello_event", String, queue_size=1)
rate = rospy.Rate(20)

# 默认动作：vx, vy, vz, yaw（前后、左右、上下、转向）
speed = 1.5
deadzone = 0.2

def scale(val):
    """将 [-1, 1] 映射为 [-speed, speed]，应用死区"""
    return int(val * speed) if abs(val) > deadzone else 0


try:
    print("✅ 手柄控制已启动，按 A 起飞，按 B 降落，按 ZR 退出")
    while not rospy.is_shutdown():
        pygame.event.pump()
        twist = Twist()

        # 左摇杆：前后、左右（linear x/y）
        twist.linear.y = scale(joystick.get_axis(0))    # 左右
        twist.linear.x = -scale(joystick.get_axis(1))   # 前后

        # 右摇杆：上下、旋转（linear z / angular z）
        twist.angular.z = scale(joystick.get_axis(2))   # 旋转
        twist.linear.z = -scale(joystick.get_axis(3))   # 上下

        pub.publish(twist)
        
        if joystick.get_button(1):	# A键：起飞
            event_pub.publish("takeoff")
        if joystick.get_button(0):	# B键：降落
            event_pub.publish("land")
        if joystick.get_button(9):	# ZR键：退出
            print("⛔ 手动退出")
            break              

        rate.sleep()

except KeyboardInterrupt:
    print("🛑 Ctrl+C 中断")


finally:
    pub.publish(Twist())
    pygame.quit()
    print("✅ 已退出手柄控制")
    
