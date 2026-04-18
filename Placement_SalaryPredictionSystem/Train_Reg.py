import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train_reg():
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment('Streamlit-Pipeline-Reg')

    train_data = pd.read_csv('artifacts/train_regression.csv')
    x_train = train_data.drop(['salary_lpa'], axis=1)
    y_train = train_data['salary_lpa']

    with mlflow.start_run(run_name='GBM_Reg') as run:
        model = GradientBoostingRegressor(random_state=42)
        
        model.fit(x_train, y_train)

        y_pred = model.predict(x_train)
        MAE = mean_absolute_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path='reg-model',
            registered_model_name='GBM_Reg_Model'
        )

        mlflow.log_metric('MAE_Score', MAE)
        mlflow.log_metric('r2_Score', r2)

        os.makedirs('artifacts', exist_ok=True)
        joblib.dump(model, 'artifacts/model_reg.pkl')
        run_id = run.info.run_id
        print(f'Training selesai. Run ID: {run_id}')

        return run_id
if __name__ == '__main__':
    train_reg()