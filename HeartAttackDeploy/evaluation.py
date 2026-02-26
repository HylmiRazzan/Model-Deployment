import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import recall_score
from pathlib import Path

BASE_DIR = Path(__file__).parent
PROCESSED_DIR = BASE_DIR

def evaluate(run_id):
    test_data = pd.read_csv(PROCESSED_DIR / "test.csv")

    x_test = test_data.drop("target", axis=1)
    y_test = test_data["target"]

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/heart-disease-model")

    predictions = model.predict(x_test)
    Recall = recall_score(y_test, predictions)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("Recall", Recall)

    print(f"Evaluation completed | Recall = {Recall:.3f}")

    return Recall

