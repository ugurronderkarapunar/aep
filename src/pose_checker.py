import streamlit as st
import numpy as np

# RTMlib'i projene ekle
try:
    from rtmlib import RTMDet, Body, draw_skeleton
except ImportError:
    st.error("RTMlib yüklenemedi. Lütfen terminalden 'pip install rtmlib' komutunu çalıştırın.")
    st.stop()

# Modeli bir kere yükle (performans için)
@st.cache_resource
def load_pose_model():
    # OpenCV'nin yerini alan, hafif bir pose modeli
    body_model = Body(
        model='rtmpose-m',
        device='cpu',  # Streamlit Cloud'da GPU olmadığı için CPU
        backend='onnxruntime' # ONNX ile daha hızlı çalışır
    )
    return body_model

def calculate_angle(a, b, c):
    """3 nokta arasındaki açıyı hesaplar."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def check_squat_form():
    """Squat hareketi için form kontrolü (kamerayı kullanır)."""
    # Bu fonksiyon, RTMlib ile çalışacak şekilde yeniden yazılmalıdır.
    # Detaylı implementasyon için özel bir rehber gerekebilir.
    st.info("Bu özellik RTMlib ile uyumlu hale getiriliyor. Lütfen daha sonra tekrar deneyin.")
    return False

def check_pushup_form():
    """Şınav hareketi için form kontrolü (kamerayı kullanır)."""
    st.info("Bu özellik RTMlib ile uyumlu hale getiriliyor. Lütfen daha sonra tekrar deneyin.")
    return False
