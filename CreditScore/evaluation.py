import mlflow
import mlflow.sklearn
import pandas as pd
from typing import Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelEvaluator:
    """Handles pulling the pipeline back out from MLflow tracking and validating it."""
    
    def run(self, run_id: str, model_name: str, x_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, float, float, float]:
        print(f"\n--- Step: Evaluation for {model_name.upper()} ---")
        
        # Mengambil model dari MLflow menggunakan run_id dan nama artefak yang sesuai
        model_uri = f"runs:/{run_id}/model_{model_name}"
        
        # Jujur saja, jika model_uri salah nama, proses ini akan gagal. 
        # Oleh karena itu, pastikan model_name sama persis dengan yang ada di train.py
        model = mlflow.sklearn.load_model(model_uri)

        # Melakukan prediksi pada data testing
        preds = model.predict(x_test)
        
        # Menghitung metrik (menggunakan zero_division=0 agar program tidak crash)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="macro", zero_division=0)
        rec = recall_score(y_test, preds, average="macro", zero_division=0)
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)

        # Membuka kembali sesi MLflow yang sedang berjalan untuk mencatat hasil evaluasi
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)

        print(f"Evaluasi selesai | Accuracy = {acc:.3f} | Precision = {prec:.3f} | Recall = {rec:.3f} | F1-Score = {f1:.3f}")
        
        return acc, prec, rec, f1