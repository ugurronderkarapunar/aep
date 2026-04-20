import cv2
import mediapipe as mp
import streamlit as st
from src.utils import calculate_angle

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def check_squat_form():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.warning("Kamera açılamadı. Form kontrolü atlanıyor.")
        return False
    
    stframe = st.empty()
    correct_count = 0
    required_frames = 20
    instruction = st.empty()
    
    for i in range(required_frames):
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            # Sol bacak açısı
            hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
            ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
            angle = calculate_angle(hip, knee, ankle)
            # Squat için ideal açı 70-110 derece
            if 70 < angle < 110:
                correct_count += 1
                instruction.info("✅ İyi squat!")
            else:
                instruction.warning("📐 Diz açını 90 dereceye yaklaştır.")
        else:
            instruction.warning("Vücudunu tamamen göster, lütfen.")
        
        # Görüntüyü göster
        stframe.image(frame, channels="BGR")
    
    cap.release()
    return correct_count > (required_frames * 0.6)  # %60 başarı
