import cv2
import mediapipe as mp
import streamlit as st

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def check_squat_form():
    cap = cv2.VideoCapture(0)  # Telefon kamerası için mobilde farklı index
    stframe = st.empty()
    correct_count = 0
    for _ in range(30):  # 1 saniye kontrol
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            # Diz ve kalça açısı hesapla
            hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
            ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
            angle = calculate_angle(hip, knee, ankle)
            if 70 < angle < 110:
                correct_count += 1
        stframe.image(frame, channels="BGR")
    cap.release()
    return correct_count > 20
