with mlflow.start_run(run_name="GBM_HeartDisease_Tuning"):
    
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
            registered_model_name="GBM_Heart_Model"
        )

        mlflow.log_metric("best_recall_score", grid_gb.best_score_)