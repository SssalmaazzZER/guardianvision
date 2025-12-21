import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# ==========================================
# ⚙️ SETUP
# ==========================================
# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Cannot open camera.")
    exit()

# File Name
file_name = "training_data.csv"
file_exists = os.path.isfile(file_name)

# ==========================================
# 📝 CSV PREPARATION
# ==========================================
# Open CSV in 'append' mode ('a') so we don't delete old data
f = open(file_name, 'a', newline='')
writer = csv.writer(f)

# If it's a new file, write the header row
if not file_exists:
    # Header: class, x1, y1, z1, v1, x2, y2, ... (for all 33 landmarks)
    landmarks = ['class']
    for val in range(1, 34):
        landmarks += [f'x{val}', f'y{val}', f'z{val}', f'v{val}']
    writer.writerow(landmarks)

# ==========================================
# 🎥 RECORDING LOOP
# ==========================================
print("---------------------------------------")
print("📷 RECORDER STARTED!")
print("---------------------------------------")
print("Hold '1' : Record STANDING")
print("Hold '2' : Record WALKING")
print("Hold '3' : Record SITTING")
print("Hold '4' : Record LYING DOWN (Fall)")
print("Press 'q': QUIT")
print("---------------------------------------")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Process Frame (Convert to RGB for AI)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # 2. Extract Data if Body is Visible
    if results.pose_landmarks:
        # Draw Skeleton on screen
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        # Create the row of data [x1, y1, z1, v1, x2...]
        row = []
        for lm in results.pose_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z, lm.visibility])
        
        # 3. Listen for Keys (Hold to Record)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('1'):
            writer.writerow(['Standing'] + row)
            print("✅ Saving STANDING...")
            
        elif key == ord('2'):
            writer.writerow(['Walking'] + row)
            print("✅ Saving WALKING...")
            
        elif key == ord('3'):
            writer.writerow(['Sitting'] + row)
            print("✅ Saving SITTING...")
            
        elif key == ord('4'):
            writer.writerow(['Lying Down'] + row)
            print("✅ Saving LYING DOWN...")
            
        elif key == ord('q'):
            print("🛑 Quitting...")
            break

    # Show the video feed
    cv2.imshow('Data Recorder (Hold 1, 2, 3, or 4)', frame)

# Cleanup
cap.release()
cv2.destroyAllWindows()
f.close()
print("💾 Data saved successfully to 'training_data.csv'")