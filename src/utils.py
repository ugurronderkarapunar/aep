import sqlite3
from datetime import datetime
import numpy as np

def init_database():
    conn = sqlite3.connect("data/workouts.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS workouts
                 (date TEXT, user_id INTEGER, exercise_name TEXT, 
                  sets INTEGER, reps INTEGER, weight REAL, 
                  rpe INTEGER, aes_score REAL)''')
    conn.commit()
    conn.close()

def save_workout(exercise_name, sets, reps, weight, rpe, aes_score, user_id=1):
    conn = sqlite3.connect("data/workouts.db")
    c = conn.cursor()
    now = datetime.now().isoformat()
    c.execute("INSERT INTO workouts VALUES (?,?,?,?,?,?,?,?)",
              (now, user_id, exercise_name, sets, reps, weight, rpe, aes_score))
    conn.commit()
    conn.close()

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle
