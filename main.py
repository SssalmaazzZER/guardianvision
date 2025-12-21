import cv2
import mediapipe as mp
import numpy as np

# 1. Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 2. Open the Webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("✅ Camera Opened! Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # 3. Process the Frame
    # MediaPipe needs RGB, OpenCV gives BGR. We must convert.
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    # 4. Draw the Skeleton
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2), # Joints (Green)
            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)  # Bones (Red)
        )
        
 

    # 5. Show the window
    cv2.imshow('Senior Safety Camera', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()