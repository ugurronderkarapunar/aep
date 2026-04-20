import streamlit as st
import pandas as pd
from datetime import datetime
from src.data_pipeline import get_user_data, calculate_fatigue_features
from src.ml_models import FatiguePredictor, recommend_exercises
from src.pose_checker import check_squat_form  # MediaPipe ile

st.set_page_config(page_title="AFI Fitness", layout="wide")

# Mobil için responsive sidebar
with st.sidebar:
    st.image("logo.png", width=150)
    st.header("Bugün nasıl hissediyorsun?")
    sleep = st.slider("Uyku (saat)", 0, 12, 7)
    muscle_soreness = st.slider("Kas ağrısı (1-5)", 1, 5, 2)
    st.header("📊 Haftalık Trend")
    df_history = get_user_data()
    if not df_history.empty:
        st.line_chart(df_history.set_index('date')['aes_score'])

st.title("🏋️ Adaptive Fitness Intelligence")
st.markdown("Bugünkü antrenman **veri bilimi** ile kişiselleştirildi.")

# Yorgunluk adaptasyonu
features = calculate_fatigue_features(df_history)
features['sleep'] = sleep
features['soreness'] = muscle_soreness
fp = FatiguePredictor()
fatigue, factor = fp.predict(list(features.values()))
st.info(f"💪 Yorgunluk seviyen: {fatigue:.0f}/100 → Önerilen hacim faktörü: {factor:.2f}")

# Egzersiz önerisi
st.subheader("📋 Bugünkü program (Optimize Edilmiş Çeşitlilik)")
recommended_exs = recommend_exercises(df_history)
selected_ex = st.selectbox("Egzersiz seç:", recommended_exs)

# Pose estimation ile form kontrolü (isteğe bağlı)
if st.button("🎥 Şınav/Çömelme formunu kontrol et"):
    form_ok = check_squat_form()  # gerçek zamanlı kamera
    if form_ok:
        st.success("Formun doğru! 👍")
    else:
        st.warning("Sırtını düz tut, daha derine çömel.")

# Antrenman girişi
col1, col2, col3 = st.columns(3)
with col1:
    sets = st.number_input("Set", 1, 10, 3)
with col2:
    reps = st.number_input("Tekrar", 1, 50, 10)
with col3:
    weight = st.number_input("Ağırlık (kg)", 0, 200, 20)
rpe = st.slider("RPE (1-10)", 1, 10, 7)

if st.button("Antrenmanı Kaydet"):
    # Veritabanına yaz
    aes = (0.4 * (weight*reps*sets/1000)) + (0.3 * (sets*reps/20)) - (0.2 * rpe) + 0.1
    conn = sqlite3.connect("data/workouts.db")
    c = conn.cursor()
    c.execute("INSERT INTO workouts VALUES (?,?,?,?,?,?,?,?)",
              (datetime.now(), 1, selected_ex, sets, reps, weight, rpe, aes))
    conn.commit()
    conn.close()
    st.balloons()
    st.success(f"✅ Kaydedildi! Bu egzersiz için AES: {aes:.2f}")
