import pandas as pd
import sqlite3
from datetime import datetime, timedelta

def get_user_data(user_id=1):
    conn = sqlite3.connect("data/workouts.db")
    df = pd.read_sql_query("SELECT * FROM workouts WHERE user_id=? ORDER BY date", conn, params=(user_id,))
    conn.close()
    return df

def calculate_fatigue_features(df):
    # Son 7 günlük hacim, frekans, ortalama RPE
    last_week = df[df['date'] >= datetime.now() - timedelta(days=7)]
    features = {
        'total_volume': (last_week['sets'] * last_week['reps'] * last_week['weight']).sum(),
        'avg_rpe': last_week['rpe'].mean(),
        'workout_frequency': last_week.shape[0],
        'days_since_last': (datetime.now() - pd.to_datetime(df['date'].max())).days
    }
    return features
