# src/pose_transformer.py

import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import VideoTransformerBase

# MediaPipe modellerini bir kere yükleyelim
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class PoseTransformer(VideoTransformerBase):
    def __init__(self):
        # Her video karesi için pose modelini başlat
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    def calculate_angle(self, a, b, c):
        """Verilen üç nokta arasındaki açıyı (derece cinsinden) hesaplar."""
        a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def transform(self, frame):
        # Gelen kareyi RGB'ye çevir (MediaPipe RGB bekler)
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        # Eğer poz bulunursa, eklem noktalarını ve açıları çiz
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # Örnek olarak sol diz açısını hesaplayalım
            try:
                left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
                left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
                
                angle = self.calculate_angle(
                    [left_hip.x, left_hip.y],
                    [left_knee.x, left_knee.y],
                    [left_ankle.x, left_ankle.y]
                )
                
                # Açıyı ekranın üst kısmına yazdır
                cv2.putText(img, f"Sol Diz Acisi: {int(angle)}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Burada istediğiniz form kontrolünü yapabilirsiniz (örn: squat için açı 70-110 arasında mı?)
                if 70 < angle < 110:
                    cv2.putText(img, "SQUAT: DOGRU FORM!", (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(img, "SQUAT: Diz Acini 90 Dereceye Yaklastir!", (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            except:
                pass

        # İşlenmiş kareyi (BGR formatında) geri döndür
        return img
