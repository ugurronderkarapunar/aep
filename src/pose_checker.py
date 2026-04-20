import streamlit as st
import numpy as np
from rtmlib import Wholebody, draw_skeleton
import cv2  # sadece VideoCapture için, GUI yok
from src.utils import calculate_angle

# RTMlib modelini bir kere yükle (cache)
@st.cache_resource
def load_pose_model():
    # Wholebody model (vücut + yüz + eller) – hafif ve headless uyumlu
    model = Wholebody(
        to_openpose=False,
        backend='onnxruntime',  # CPU'da hızlı
        device='cpu'
    )
    return model

def get_landmarks_from_frame(model, frame):
    """Verilen kare (numpy array) üzerinden landmarkları al."""
    # frame BGR formatında olabilir, RTMlib RGB bekler
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    keypoints, scores = model(rgb)
    if keypoints is None or len(keypoints) == 0:
        return None
    # İlk kişiyi al
    kpts = keypoints[0]  # shape (n, 3) veya (n,2)
    # Kullanacağımız landmark indeksleri (COCO formatı):
    # 5: sol omuz, 6: sağ omuz, 11: sol kalça, 12: sağ kalça,
    # 13: sol diz, 14: sağ diz, 15: sol ayak, 16: sağ ayak
    return kpts

def check_squat_form():
    st.info("📷 Kamera başlatılıyor... (izin verin)")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Kamera açılamadı. Lütfen kamera izinlerini kontrol edin.")
        return False
    
    model = load_pose_model()
    stframe = st.empty()
    correct_count = 0
    required_frames = 20
    
    for _ in range(required_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Landmarkları al
        kpts = get_landmarks_from_frame(model, frame)
        if kpts is not None:
            # Sol diz açısı (kalça - diz - ayak)
            hip = kpts[11]    # sol kalça
            knee = kpts[13]   # sol diz
            ankle = kpts[15]  # sol ayak
            # Eğer landmarklar yoksa veya görünürlük düşükse geç
            if hip is not None and knee is not None and ankle is not None:
                angle = calculate_angle(hip[:2], knee[:2], ankle[:2])
                if 70 < angle < 110:
                    correct_count += 1
        
        # Görüntüyü göster (OpenCV frame BGR, Streamlit RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)
    
    cap.release()
    if correct_count > required_frames // 2:
        st.success("✅ Squat formun doğru!")
        return True
    else:
        st.warning("⚠️ Diz açını 90 dereceye yaklaştırmaya çalış.")
        return False

def check_pushup_form():
    st.info("📷 Şınav kontrolü başlatılıyor...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Kamera açılamadı.")
        return False
    
    model = load_pose_model()
    stframe = st.empty()
    correct_count = 0
    required_frames = 15
    
    for _ in range(required_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        kpts = get_landmarks_from_frame(model, frame)
        if kpts is not None:
            # Omuz - dirsek - bilek açısı (sol kol)
            shoulder = kpts[5]   # sol omuz
            elbow = kpts[7]      # sol dirsek
            wrist = kpts[9]      # sol bilek
            if shoulder is not None and elbow is not None and wrist is not None:
                angle = calculate_angle(shoulder[:2], elbow[:2], wrist[:2])
                if 90 < angle < 160:
                    correct_count += 1
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)
    
    cap.release()
    if correct_count > required_frames // 2:
        st.success("✅ Şınav formun iyi!")
        return True
    else:
        st.warning("⚠️ Dirseklerini 90 dereceye kadar kırmaya çalış.")
        return False
