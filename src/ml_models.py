import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class FatiguePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50)
    
    def train(self, X, y):
        self.model.fit(X, y)
        joblib.dump(self.model, "models/fatigue_predictor.pkl")
    
    def predict(self, features):
        model = joblib.load("models/fatigue_predictor.pkl")
        fatigue_score = model.predict([features])[0]
        # fatigue_score 0-100 arası, buna göre hacim önerisi
        recommended_volume_factor = 1.0 - (fatigue_score / 200)  # 0.5x - 1.0x
        return fatigue_score, max(0.5, min(1.1, recommended_volume_factor))

# Egzersiz önerisi için basit bir similarity modeli
def recommend_exercises(user_history_df, muscle_group_focus="full"):
    # Tüm egzersiz havuzu (kendi oluşturduğumuz CSV)
    all_exercises = pd.read_csv("data/exercise_library.csv")
    # Kullanıcının en son yaptığı 5 egzersizi al
    recent = user_history_df.sort_values('date', ascending=False).head(5)['exercise_name'].tolist()
    # Öner: daha önce yapmamış veya 7 gündür yapmamış egzersizler
    candidates = all_exercises[~all_exercises['exercise_name'].isin(recent)]
    # Kas grubu dengesine göre sırala (örneğin chest düşükse chest öne al)
    return candidates.head(5)['exercise_name'].tolist()
