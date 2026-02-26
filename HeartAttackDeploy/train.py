import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

def train():
    mlflow.set_experiment("Local-MLflow-Pipeline")

    train_data = pd.read_csv("train.csv")
    x_train = train_data.drop("target", axis=1)
    y_train = train_data["target"]

    with mlflow.start_run(run_name="GBM_HeartDisease_Tuning") as run:
    
        GBM = GradientBoostingClassifier(random_state=42)

        param_gb = {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4],
        'subsample': [0.7, 0.8, 0.9],
        'max_features': ['sqrt', 'log2']
        }

        grid_gb = GridSearchCV(
            estimator=GBM, 
            param_grid=param_gb,
            cv=5, 
            scoring='recall',
            n_jobs=-1
        )
        grid_gb.fit(x_train, y_train)

        mlflow.sklearn.log_model(
            sk_model=grid_gb.best_estimator_,
            artifact_path="heart-disease-model",
            registered_model_name="GBM_Heart_Model" #Registered model, jika ingin dismpan sebagai model terbaik, jika tidak maka hapus 1 line ini
        )

        mlflow.log_metric("best_recall_score", grid_gb.best_score_)
        run_id = run.info.run_id
        print(f"Training selesai. Run ID: {run_id}")
        return run_id
if __name__ == "__main__":
    train()

