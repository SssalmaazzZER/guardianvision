import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import time
from datetime import datetime

# ==========================================
# ⚙️ DEMO SETTINGS (Tweak these for presentation)
# ==========================================
# 1. SIMULATION MODE: Set to True to pretend it's 3 AM
SIMULATE_NIGHT = True 

# 2. SENSITIVITY
INACTIVITY_THRESHOLD = 15.0  # Seconds (If no movement for 10s -> Alert)
FALL_TIME_THRESHOLD = 3.0    # Seconds (If lying down for 3s -> Alarm)

# ==========================================
# 🧠 LOAD RESOURCES
# ==========================================
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("❌ Error: 'model.pkl' not found. Run train_model.py first!")
    exit()

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Variables for Logic
current_action = "Unknown"
last_action_time = time.time()
fall_start_time = None
room_sequence = []
last_zone = "Unknown"

# Helper: Define Zones (Left vs Right side of screen)
def detect_zone(x_px, width):
    if x_px < width / 2: return "Bedroom"
    else: return "Kitchen"

# ==========================================
# 🎥 MAIN LOOP
# ==========================================
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Image Prep
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    height, width, _ = frame.shape
    
    # Default Status
    status_color = (0, 255, 0) # Green (Safe)
    alert_msg = ""
    current_time_str = datetime.now().strftime("%H:%M:%S")

    if results.pose_landmarks:
        # Draw Skeleton
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        try:
            # 1. Extract Data
            pose_landmarks = results.pose_landmarks.landmark
            row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_landmarks]).flatten())
            X = pd.DataFrame([row])
            
            # 2. AI Prediction
            prediction = model.predict(X)[0]
            probs = model.predict_proba(X)[0]
            
            # Only trust high confidence predictions (> 50%)
            if max(probs) > 0.5:
                # Check if action changed (to reset inactivity timer)
                if prediction != current_action:
                    current_action = prediction
                    last_action_time = time.time()

            # ----------------------------------------------------
            # 🚨 THE TEACHER'S SCENARIOS (LOGIC)
            # ----------------------------------------------------
            
            # Scenario A: Fall Detection (Malaise)
            if current_action == "Lying Down" or current_action == "Fallen":
                if fall_start_time is None:
                    fall_start_time = time.time()
                elif (time.time() - fall_start_time) > FALL_TIME_THRESHOLD:
                    status_color = (0, 0, 255) # Red
                    alert_msg = "CRITICAL: FALL DETECTED!"
            else:
                fall_start_time = None

            # Scenario B: Inactivity (Absence d’activité)
            # If action is 'Sitting' or 'Standing' for too long
            if current_action != "Walking" and current_action != "Lying Down":
                elapsed = time.time() - last_action_time
                if elapsed > INACTIVITY_THRESHOLD:
                    status_color = (0, 165, 255) # Orange
                    alert_msg = f"ALERT: User Inactive ({int(elapsed)}s)"

            # Scenario C: Night Wandering (Activité nocturne)
            # If enabled, pretend it is 3 AM
            hour = 3 if SIMULATE_NIGHT else datetime.now().hour
            if (0 <= hour < 5) and current_action == "Walking":
                status_color = (0, 255, 255) # Yellow
                alert_msg = "ALERT: Night Wandering Detected"

            # Scenario D: Incoherent Zones (Bedroom <-> Kitchen)
            hip_x = pose_landmarks[24].x * width  # Right Hip
            current_zone = detect_zone(hip_x, width)
            
            # Logic: If changing rooms rapidly
            if current_zone != last_zone:
                room_sequence.append(current_zone)
                last_zone = current_zone
                if len(room_sequence) > 6: room_sequence.pop(0)
            
            # Check for pattern: A -> B -> A -> B
            if len(room_sequence) >= 4:
                # Simple check: are they bouncing back and forth?
                if room_sequence[-4:] == ['Bedroom', 'Kitchen', 'Bedroom', 'Kitchen']:
                     status_color = (255, 0, 255) # Purple
                     alert_msg = "ALERT: Disoriented Pacing"

        except Exception as e:
            pass
    
    # ----------------------------------------------------
    # 🖥️ DASHBOARD UI
    # ----------------------------------------------------
    # Top Bar Background
    cv2.rectangle(image, (0,0), (640, 80), (245, 117, 16), -1)
    
    # Class
    cv2.putText(image, 'ACTION', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(image, current_action, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
    # Zone
    cv2.putText(image, 'ZONE', (250,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(image, last_zone, (245,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    # Time
    cv2.putText(image, 'TIME', (480,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    t_str = "03:00 (SIM)" if SIMULATE_NIGHT else current_time_str
    cv2.putText(image, t_str, (450,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    # FLASHING ALERT
    if alert_msg:
        # Draw red border
        cv2.rectangle(image, (0,0), (width, height), status_color, 10)
        # Draw message box
        cv2.rectangle(image, (0, 200), (640, 300), status_color, -1)
        cv2.putText(image, alert_msg, (20, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow('GuardianVision - Senior Safety', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()