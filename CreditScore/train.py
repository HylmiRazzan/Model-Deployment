from pathlib import Path
from typing import Tuple
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Pastikan file preprocess.py berada di folder yang sama
from preprocess import CreditPreprocessor


class ModelTrainer:
    """Handles feature definition, pipeline building, training, and artifact tracking for multiple models."""
    
    def __init__(self, experiment_name: str = "Credit Score Classification", 
                 artifact_path: str = "artifacts", random_state: int = 42):
        self.experiment_name = experiment_name
        self.artifact_dir = Path(artifact_path)
        self.preprocessor = CreditPreprocessor()
        self.random_state = random_state
        
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_experiment(self.experiment_name)
        
        # ---------------------------------------------------------
        # DICTIONARY MODEL DIDEFINISIKAN DI DALAM CONSTRUCTOR
        # ---------------------------------------------------------
        self.models = {
            "random_forest": {
                "instance": RandomForestClassifier(
                    random_state=self.random_state, n_jobs=-1, n_estimators=300,
                    max_depth=16, min_samples_leaf=5, max_features=0.5, criterion="gini"
                ),
                "params": {
                    "n_estimators": 300, "max_depth": 16, "min_samples_leaf": 5, 
                    "max_features": 0.5, "criterion": "gini"
                }
            },
            "lightgbm": {
                "instance": LGBMClassifier(
                    random_state=self.random_state, n_jobs=-1, verbose=-1,
                    n_estimators=300, max_depth=6, num_leaves=40, learning_rate=0.05,
                    min_child_samples=30, subsample=0.8, colsample_bytree=0.8
                ),
                "params": {
                    "n_estimators": 300, "max_depth": 6, "num_leaves": 40, 
                    "learning_rate": 0.05, "min_child_samples": 30, "subsample": 0.8, 
                    "colsample_bytree": 0.8
                }
            },
            "xgboost": {
                "instance": XGBClassifier(
                    random_state=self.random_state, n_jobs=-1, objective='multi:softmax',
                    eval_metric='mlogloss', n_estimators=500, max_depth=6, gamma=1, 
                    min_child_weight=2, subsample=0.8, learning_rate=0.1, colsample_bytree=0.8
                ),
                "params": {
                    "objective": 'multi:softmax', "eval_metric": 'mlogloss', 
                    "n_estimators": 500, "max_depth": 6, "gamma": 1, "min_child_weight": 2, 
                    "subsample": 0.8, "learning_rate": 0.1, "colsample_bytree": 0.8
                }
            }
        }

    def run(self, data_path: str | Path, model_name: str) -> Tuple[str, pd.DataFrame, pd.Series]:
        """
        Menjalankan pipeline training.
        Parameter `model_name` harus salah satu dari: 'random_forest', 'lightgbm', 'xgboost'
        """
        # Validasi nama model
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' tidak ditemukan. Pilihan yang tersedia: {list(self.models.keys())}")

        x_train, x_test, y_train, y_test = self.preprocessor.clean_and_split(data_path)
        transformer = self.preprocessor.get_transformer(x_train)
        
        # Ambil instance model dan parameternya dari dictionary
        selected_model = self.models[model_name]["instance"]
        selected_params = self.models[model_name]["params"]
        
        # Susun Pipeline
        final_pipeline = ImbPipeline(steps=[
            ('preprocess', transformer),
            ('smote', SMOTE(random_state=self.random_state)),
            ('classifier', selected_model) # Masukkan model yang dipilih ke ujung pipeline
        ])

        # Gunakan run_name agar di UI MLflow terlihat jelas model apa yang sedang dilatih
        with mlflow.start_run(run_name=model_name) as run:
            
            # Catat parameter spesifik milik model tersebut
            mlflow.log_params(selected_params)

            # Latih model
            final_pipeline.fit(x_train, y_train)

            # Simpan file dan catat artefak menggunakan nama yang dinamis
            model_file_path = self.artifact_dir / f"score_prediction_{model_name}.pkl"
            joblib.dump(final_pipeline, model_file_path)
            mlflow.sklearn.log_model(final_pipeline, artifact_path=f"model_{model_name}")
            
            print(f"[{model_name.upper()}] Berhasil dilatih dan disimpan ke {model_file_path}")
            return run.info.run_id, x_test, y_test