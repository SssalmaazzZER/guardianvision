import cv2
from ultralytics import YOLO
import time

# --- 1. CHARGEMENT DU MODÈLE ---
# On utilise 'yolov8n-pose.pt'. 
# 'n' signifie 'nano' (le plus rapide pour CPU). 
# 'pose' signifie qu'il est entraîné pour le squelette.
print("Chargement du modèle YOLOv8-Pose...")
model = YOLO('yolov8n-pose.pt') 

# --- 2. INITIALISATION VIDÉO ---
cap = cv2.VideoCapture(0)

# Variables FPS
prev_frame_time = 0
new_frame_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- 3. DÉTECTION MULTI-PERSONNES ---
    # conf=0.5 : On ne garde que les détections sûres à 50%
    # results contient une liste de TOUTES les personnes détectées
    results = model(frame, conf=0.5, verbose=False)

    # --- 4. VISUALISATION ---
    # YOLO a une fonction intégrée 'plot()' qui dessine tout (boites + squelettes)
    # pour toutes les personnes détectées.
    annotated_frame = results[0].plot()

    # --- 5. FPS (Performance) ---
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Affichage
    cv2.imshow('GuardianVision - Multi-Person Detection', annotated_frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()