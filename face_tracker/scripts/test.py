import cv2
import mediapipe as mp

img = cv2.imread("TY_face.png")
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)
    
    if results.detections:
        print(f"Detected {len(results.detections)} face(s)")
        for detection in results.detections:
            mp_draw.draw_detection(img, detection)
    else:
        print("No face detected.")

cv2.imshow("Mediapipe Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

