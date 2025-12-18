"""
PROJET : GuardianVision
MODULE : Phase 1 - Perception & Extraction Squelettique
AUTEUR : [Ton Nom]
DESCRIPTION : 
Ce script capture le flux vidéo en temps réel, utilise MediaPipe (Google) 
pour extraire 33 points clés du corps humain (Landmarks) et affiche le résultat.
Il sert de base pour l'analyse comportementale (Phase 2 & 3).
"""

import cv2
import mediapipe as mp
import numpy as np
import time

# --- 1. CONFIGURATION & INITIALISATION ---

# Initialisation des modules de dessin (pour la visualisation)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Configuration de la fenêtre vidéo
window_name = "GuardianVision - Phase 1: Real-Time Perception"

# Initialisation de la capture vidéo (0 = Webcam par défaut)
cap = cv2.VideoCapture(0)

# Variables pour calculer les FPS (Performance)
prev_frame_time = 0
new_frame_time = 0

# --- 2. BOUCLE PRINCIPALE (MOTEUR DE PERCEPTION) ---

# On utilise le contexte "with" pour gérer la mémoire proprement
# min_detection_confidence=0.5 : Il faut être sûr à 50% que c'est une personne pour détecter
# min_tracking_confidence=0.5 : Il faut être sûr à 50% pour suivre le mouvement (évite le tremblement)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Sécurité : Si la caméra ne renvoie rien, on arrête
        if not ret:
            print("Erreur : Impossible de lire le flux vidéo.")
            break

        # A. PRÉ-TRAITEMENT (Optimisation)
        # MediaPipe a besoin de RGB, OpenCV donne du BGR. On convertit.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Pour gagner en performance, on marque l'image comme "non modifiable" 
        # avant de la passer au modèle (pass-by-reference)
        image.flags.writeable = False
      
        # B. DÉTECTION (Le cœur de l'IA)
        # C'est ici que les pixels deviennent des données mathématiques
        results = pose.process(image)
    
        # C. POST-TRAITEMENT
        # On rend l'image modifiable pour dessiner dessus
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # D. EXTRACTION & VISUALISATION
        if results.pose_landmarks:
            # 1. Dessiner le squelette sur l'image
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), # Articulations (Orange)
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)  # Os (Rose)
            )
            
            # 2. (Optionnel) Extraire les coordonnées brutes pour vérifier
            # Exemple : Récupérer la position du NEZ (Index 0)
            try:
                nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                # On affiche les coordonnées dans la console pour prouver au prof qu'on a des données
                # print(f"Nez -> x: {nose.x}, y: {nose.y}, z: {nose.z}") 
            except:
                pass

        # E. CALCUL DES FPS (Preuve de fluidité)
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps_text = f"FPS: {int(fps)}"

        # Afficher les FPS sur l'écran
        cv2.putText(image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Afficher le résultat final
        cv2.imshow(window_name, image)

        # Touche 'q' pour quitter proprement
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# --- 3. FERMETURE PROPRE ---
cap.release()
cv2.destroyAllWindows()
print("Programme terminé avec succès.")