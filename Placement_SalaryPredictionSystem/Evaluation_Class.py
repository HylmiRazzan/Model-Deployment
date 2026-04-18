import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import f1_score
from pathlib import Path

BASE_DIR = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "artifacts"

def evaluate_class(run_id):
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment('Streamlit-Pipeline-Clas')
    test_data = pd.read_csv(PROCESSED_DIR / 'test_classification.csv')

    x_test = test_data.drop(['placement_status'], axis=1)
    y_test = test_data['placement_status']

    model = mlflow.sklearn.load_model(f'runs:/{run_id}/class-model')

    predictions = model.predict(x_test)
    F1 = f1_score(y_test, predictions)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric('F1_Score', F1)

    print(f'Evaluasi Selesai | F1_Score = {F1:.3f}')

    return F1