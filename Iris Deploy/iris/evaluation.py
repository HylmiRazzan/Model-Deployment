import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from pathlib import Path

BASE_DIR = Path(__file__).parent
PROCESSED_DIR = BASE_DIR #/ "data" / "processed"

def evaluate(test_scaled,run_id):

    X_test = test_scaled.drop("species", axis=1)
    y_test = test_scaled["species"]

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro")
    rec = recall_score(y_test, preds, average="macro")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

    print(f"Evaluation completed | Accuracy = {acc:.3f}")

    return acc, prec,rec

if __name__ == "__main__":
    evaluate()



