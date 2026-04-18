import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

def train_clas():
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment('Streamlit-Pipeline-Clas')

    train_data = pd.read_csv('artifacts/train_classification.csv')

    x_train = train_data.drop(['placement_status'], axis = 1)
    y_train = train_data['placement_status']

    with mlflow.start_run(run_name='XGB_Clas') as run:
        model = XGBClassifier(
                    eval_metric='aucpr',
                    random_state=42,
                    use_label_encoder=False
                )
        
        model.fit(x_train, y_train)

        y_pred = model.predict(x_train)
        f1 = f1_score(y_train, y_pred)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path='class-model',
            registered_model_name='XGB_Class_Model'
        )

        mlflow.log_metric('F1_Score', f1)

        os.makedirs('artifacts', exist_ok=True)
        joblib.dump(model, 'artifacts/model_clas.pkl')
        run_id = run.info.run_id
        print(f'Training selesai. Run ID: {run_id}')

        return run_id

if __name__ == '__main__':
    train_clas()