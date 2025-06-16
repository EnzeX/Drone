#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
from collections import deque

# 初始化
bridge = CvBridge()
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# 控制参数
pid = [0.2, 0.1]  # P, D
detect_range = [3000, 15000]
prev_err_rotate = 0
prev_err_up = 0
last_process_time = 0
lost_counter = 0

# 控制发布器
cmd_pub = None

# 平滑处理队列
window_size = 5
face_center_x_q = deque(maxlen=window_size)
face_center_y_q = deque(maxlen=window_size)
area_q = deque(maxlen=window_size)

def smooth(val_q):
    return sum(val_q) // len(val_q) if val_q else 0

def publish_cmd(fb, up, yaw):
    cmd = Twist()
    cmd.linear.x = fb / 10.0
    cmd.linear.z = up / 10.0
    cmd.angular.z = yaw / 30.0
    cmd_pub.publish(cmd)

def find_face(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    if results.detections:
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        h, w = img.shape[:2]
        cx = int((bboxC.xmin + bboxC.width / 2) * w)
        cy = int((bboxC.ymin + bboxC.height / 2) * h)
        area = int(bboxC.width * bboxC.height * w * h)

        # 平滑
        face_center_x_q.append(cx)
        face_center_y_q.append(cy)
        area_q.append(area)

        cx = smooth(face_center_x_q)
        cy = smooth(face_center_y_q)
        area = smooth(area_q)

        # 画边界框
        x1 = int(bboxC.xmin * w)
        y1 = int(bboxC.ymin * h)
        x2 = int((bboxC.xmin + bboxC.width) * w)
        y2 = int((bboxC.ymin + bboxC.height) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

        return [cx, cy], area
    else:
        return [0, 0], 0

def image_cb(msg):
    global prev_err_rotate, prev_err_up, last_process_time, lost_counter
    now = rospy.get_time()
    if now - last_process_time < 0.05:
        return
    last_process_time = now

    img = bridge.imgmsg_to_cv2(msg, "bgr8")
    h, w = img.shape[:2]
    info, area = find_face(img)
    x, y = info

    err_rotate = x - w // 2
    err_up = h // 2 - y

    # 死区
    if abs(err_rotate) < 25:
        err_rotate = 0
    if abs(err_up) < 20:
        err_up = 0

    rotate_speed = int(pid[0] * err_rotate + pid[1] * (err_rotate - prev_err_rotate))
    up_speed = -int(pid[0] * err_up + pid[1] * (err_up - prev_err_up))
    rotate_speed = max(min(rotate_speed, 20), -20)
    up_speed = max(min(up_speed, 20), -20)

    # 前后速度
    if detect_range[0] < area < detect_range[1]:
        fb = 0
    elif area > detect_range[1]:
        fb = -20
    elif area < detect_range[0] and area > 1000:
        fb = 10
    else:
        fb = 0

    # 控制逻辑
    if x == 0:
        lost_counter += 1
        if lost_counter >= 5:
            publish_cmd(0, 0, 0)
    else:
        lost_counter = 0
        publish_cmd(fb, up_speed, rotate_speed)

    prev_err_rotate = err_rotate
    prev_err_up = err_up

    # 可视化
    cv2.circle(img, (w // 2, h // 2), 5, (0, 255, 0), -1)  # 图像中心
    if x != 0:
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)         # 人脸中心
        cv2.arrowedLine(img, (w//2, h//2), (x, y), (0, 0, 255), 2)
    cv2.putText(img, f"Area: {area}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
    cv2.imshow("Face Tracking", img)
    cv2.waitKey(1)

def main():
    global cmd_pub
    rospy.init_node("face_tracker_mediapipe")
    cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    rospy.Subscriber("/airsim_node/drone1/front_center/Scene", Image, image_cb)
    rospy.spin()

if __name__ == "__main__":
    main()

