import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from datetime import datetime
from src.data_pipeline import get_user_data, calculate_fatigue_features
from src.ml_models import FatiguePredictor, recommend_exercises
from src.pose_checker import check_squat_form, check_pushup_form
from src.utils import init_database, save_workout

init_database()

st.set_page_config(page_title="AFI Fitness", layout="wide")

with st.sidebar:
    st.title("🏋️ AFI")
    sleep = st.slider("Uyku (saat)", 0, 12, 7)
    muscle_soreness = st.slider("Kas ağrısı (1-5)", 1, 5, 2)
    st.markdown("---")
    df_history = get_user_data()
    if not df_history.empty:
        st.subheader("📈 AES Trend")
        st.line_chart(df_history.set_index('date')['aes_score'])

st.title("💪 Adaptive Fitness Intelligence")
st.markdown("Veri bilimi ile kişiselleştirilmiş antrenman (RTMlib ile form analizi)")

df_history = get_user_data()
features_base = calculate_fatigue_features(df_history)
features = [features_base['total_volume'], features_base['avg_rpe'],
            features_base['workout_frequency'], features_base['days_since_last'],
            sleep, muscle_soreness]
fp = FatiguePredictor()
fatigue, factor = fp.predict(features)
st.info(f"🧠 Yorgunluk: {fatigue:.0f}/100 → Önerilen hacim faktörü: {factor:.2f}")

st.subheader("📋 Bugünün önerilen egzersizleri")
recommended = recommend_exercises(df_history)
selected_ex = st.selectbox("Egzersiz seç", recommended)

col1, col2 = st.columns(2)
with col1:
    if st.button("🎥 Squat formunu kontrol et"):
        with st.spinner("Kamera açılıyor, poz ver..."):
            check_squat_form()
with col2:
    if st.button("🎥 Şınav formunu kontrol et"):
        with st.spinner("Kamera açılıyor..."):
            check_pushup_form()

st.subheader("✍️ Antrenman kaydet")
sets = st.number_input("Set", 1,10,3)
reps = st.number_input("Tekrar",1,50,10)
weight = st.number_input("Ağırlık (kg)",0.0,200.0,20.0)
rpe = st.slider("RPE (1-10)",1,10,7)

if st.button("💾 Kaydet"):
    volume_norm = min((sets*reps*weight)/5000, 1.0)
    aes = 0.4*volume_norm + 0.3*(sets*reps/50) - 0.2*(rpe/10) + 0.1
    aes = max(0, min(1, aes))
    save_workout(selected_ex, sets, reps, weight, rpe, aes)
    st.balloons()
    st.success(f"Kaydedildi! AES: {aes:.2f}")
    st.rerun()

st.subheader("📜 Geçmiş antrenmanlar")
if not df_history.empty:
    st.dataframe(df_history[['date','exercise_name','sets','reps','weight','rpe','aes_score']].tail(10))
else:
    st.info("Henüz kayıtlı antrenman yok.")
