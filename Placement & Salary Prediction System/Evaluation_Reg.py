import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path

BASE_DIR = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "artifacts"

def evaluate_reg(run_id):
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment('Streamlit-Pipeline-Reg')
    test_data = pd.read_csv(PROCESSED_DIR / 'test_regression.csv')

    x_test = test_data.drop(['salary_lpa'], axis=1)
    y_test = test_data['salary_lpa']

    model = mlflow.sklearn.load_model(f'runs:/{run_id}/reg-model')

    predictions = model.predict(x_test)
    MAE = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric('MAE_Score', MAE)
        mlflow.log_metric('r2_Score', r2)

    print(f'Evaluasi Selesai | MAE_Score = {MAE:.3f} | r2_Score = {r2:.3f}')

    return MAE, r2