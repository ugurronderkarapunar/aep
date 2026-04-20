import sys
import os
from pathlib import Path

# Bu satır, src klasörünün bulunmasını sağlar
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
from datetime import datetime
import sqlite3
import plotly.express as px

# Kendi modüllerimiz (artık src bulunacak)
from src.data_pipeline import get_user_data, calculate_fatigue_features
from src.ml_models import FatiguePredictor, recommend_exercises
from src.pose_checker import check_squat_form
from src.utils import init_database, save_workout

# Sayfa yapılandırması
st.set_page_config(page_title="AFI Fitness", layout="wide", initial_sidebar_state="auto")

# Veritabanını başlat
init_database()

# Mobil için sidebar
with st.sidebar:
    st.image("https://img.icons8.com/ios-filled/100/000000/dumbbell.png", width=80)
    st.header("🏋️ Bugün nasıl hissediyorsun?")
    sleep = st.slider("😴 Uyku (saat)", 0, 12, 7)
    muscle_soreness = st.slider("💪 Kas ağrısı (1-5)", 1, 5, 2)
    st.header("📊 Haftalık Trend")
    df_history = get_user_data()
    if not df_history.empty:
        # Son 7 günlük AES skoru grafiği
        df_week = df_history.tail(7)
        fig = px.line(df_week, x='date', y='aes_score', title='Antrenman Etkinlik Skoru')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Henüz antrenman verisi yok.")

st.title("🧠 Adaptive Fitness Intelligence")
st.markdown("Veri bilimi ile kişiselleştirilmiş antrenman deneyimi")

# Yorgunluk adaptasyonu
if not df_history.empty:
    features_dict = calculate_fatigue_features(df_history)
    features = [features_dict['total_volume'], features_dict['avg_rpe'], 
                features_dict['workout_frequency'], features_dict['days_since_last'], 
                sleep, muscle_soreness]
    fp = FatiguePredictor()
    fatigue, factor = fp.predict(features)
    st.info(f"💪 **Yorgunluk seviyen:** {fatigue:.0f}/100  →  Önerilen hacim faktörü: **{factor:.2f}**")
else:
    st.info("İlk antrenmanını kaydetmek için aşağıdaki formu doldur.")
    fatigue = 30
    factor = 0.9

# Egzersiz önerisi
st.subheader("📋 Bugünkü program (Optimize Edilmiş Çeşitlilik)")
recommended_exs = recommend_exercises(df_history)
selected_ex = st.selectbox("Egzersiz seç:", recommended_exs)

# Pose estimation ile form kontrolü (opsiyonel)
if st.button("🎥 Form kontrolü (squat/push-up)"):
    with st.spinner("Kamera açılıyor, lütfen bekleyin..."):
        form_ok = check_squat_form()
    if form_ok:
        st.success("✅ Formun doğru! Tebrikler.")
    else:
        st.warning("⚠️ Formunda iyileştirme gerekebilir. Sırtını düz tut ve derine çömel.")

# Antrenman girişi
st.subheader("✍️ Antrenmanını kaydet")
col1, col2, col3 = st.columns(3)
with col1:
    sets = st.number_input("Set sayısı", min_value=1, max_value=10, value=3)
with col2:
    reps = st.number_input("Tekrar sayısı", min_value=1, max_value=50, value=10)
with col3:
    weight = st.number_input("Ağırlık (kg)", min_value=0, max_value=200, value=20, step=5)
rpe = st.slider("RPE (1-10) – Zorluk derecesi", 1, 10, 7)

if st.button("💾 Antrenmanı kaydet", type="primary"):
    # Antrenman Etkinlik Skoru (AES) hesapla
    # Hacim (set*rep*weight) normalize edilmiş (0-1 arası kabaca)
    volume_norm = min(1.0, (sets * reps * weight) / 3000)
    hacim_norm = min(1.0, (sets * reps) / 50)
    aes = (0.4 * volume_norm) + (0.3 * hacim_norm) - (0.2 * (rpe/10)) + 0.1
    aes = max(0, min(1, aes)) * 10  # 0-10 arası skor
    save_workout(selected_ex, sets, reps, weight, rpe, aes)
    st.balloons()
    st.success(f"✅ Kaydedildi! Bu egzersiz için AES: {aes:.2f}/10")
    st.rerun()
