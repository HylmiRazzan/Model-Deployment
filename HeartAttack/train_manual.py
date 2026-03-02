import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier # Ganti jika pakai model lain
from sklearn.preprocessing import StandardScaler

# 1. Load Data
try:
    df = pd.read_csv("Heart Attack Data Set.csv") # Pastikan file heart.csv ada di folder yang sama
    X = df.drop('target', axis=1)
    y = df['target']


    # 3. Training (Gunakan model yang sama dengan tugas Anda)
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 4. Save dengan "Bahasa" Python 3.13 (Scikit-Learn 1.6.0)
    joblib.dump(model, 'model.pkl')

    print("--- BERHASIL! ---")
    print("File model.pkl dan preprocessing.pkl versi 1.6.0 sudah siap.")
except Exception as e:
    print(f"Gagal: {e}")