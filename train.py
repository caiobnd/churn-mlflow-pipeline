from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from xgboost                import XGBClassifier
from sklearn.metrics        import recall_score,f1_score
from pathlib                import Path
from joblib                 import dump
import cleanning
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
path_data=Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df          = cleanning.load_data(path_data)
df_clean    = cleanning.clean_data(df)
df_enconded = cleanning.encoding(df_clean)
X_train, X_test, y_train, y_test = cleanning.split_data(df_enconded)

models_1 = {
    'xgb_v1': XGBClassifier(n_estimators=100, learning_rate=0.1, scale_pos_weight=2.77, random_state=42),
    'xgb_v2': XGBClassifier(n_estimators=200, learning_rate=0.05, scale_pos_weight=2.77, random_state=42),
    'rf_v1':  RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'lr_v1':  LogisticRegression(class_weight='balanced', max_iter=2000)
}

models_2 = {'lr_v2': LogisticRegression(class_weight='balanced', max_iter=2000, solver='saga'),
            'xgb_v3': XGBClassifier(n_estimators=300, learning_rate=0.05, scale_pos_weight=2.77, random_state=42),
            'xgb_v4': XGBClassifier(n_estimators=100, learning_rate=0.1, scale_pos_weight=3.5, random_state=42),
            'rf_v2':  RandomForestClassifier(n_estimators=200, class_weight='balanced', max_depth=10, random_state=42),}

#apenas para saciar a dúvida
models_3={'lr_v3':  LogisticRegression(class_weight='balanced', max_iter=3000,solver='saga')}

models = {**models_1,**models_2,**models_3}

for name,model in models_1.items():
    with mlflow.start_run(run_name=name):
        mlflow.log_params(model.get_params())
        model_path      = Path(f"model/{name}.pkl")
        trained_model   = model.fit(X_train,y_train)
        y_pred = trained_model.predict(X_test)
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        dump(trained_model,model_path)
        mlflow.log_artifact(str(model_path))
        
    
