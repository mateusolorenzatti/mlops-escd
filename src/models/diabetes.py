import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.under_sampling import RandomUnderSampler
from mlflow.models.signature import infer_signature

import xgboost as xgb

mlflow.set_tracking_uri("sqlite:///../../mlflow.db")
mlflow.set_experiment("diabetes_experiment")

# Carregar o dataset
def load_data():
    df = pd.read_csv('../../data/diabetes_binary.csv')
    features = ['HighBP', 'HighChol', 'BMI', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity',
                'HvyAlcoholConsump', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Age',
                'Income']
    target = 'Diabetes_binary'
    return df[features], df[target]

# Preprocessamento dos dados
def preprocess_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    undersampler = RandomUnderSampler(random_state=42)
    X_train, y_train = undersampler.fit_resample(X_train, y_train)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# Treinar e avaliar Random Forest
def train_random_forest(X_train, X_test, y_train, y_test):

    with mlflow.start_run(run_name="RandomForest"):
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        signature = infer_signature(X_train, y_pred)
        model_info = mlflow.sklearn.log_model(rf_model, "random_forest_model", 
                                    signature=signature, 
                                    input_example=X_train, 
                                    registered_model_name="RandomForest")
        
        loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
        predictions = loaded_model.predict(X_test)
        result = pd.DataFrame(X_test, columns=X.columns.values)
        result["label"] = y_test.values
        result["predictions"] = predictions

        mlflow.evaluate(
            data=result,
            targets="label",
            predictions="predictions",
            model_type="classifier",
        )
        
        print('Random Forest Results:')
        print(f'Acurácia: {accuracy:.4f}')
        print(f'Precisão: {precision:.4f}')
        print(f'Revocação: {recall:.4f}')
        print(f'F1-score: {f1:.4f}')

# Treinar e avaliar XGBoost
def train_xgboost(X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name="XGBoost"):
        xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        signature = infer_signature(X_train, y_pred)
        model_info = mlflow.sklearn.log_model(xgb_model, "xgboost_model", 
                                    signature=signature,    
                                    input_example=X_train, 
                                    registered_model_name="XGBoost")
    
        loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
        predictions = loaded_model.predict(X_test)
        result = pd.DataFrame(X_test, columns=X.columns.values)
        result["label"] = y_test.values
        result["predictions"] = predictions

        mlflow.evaluate(
            data=result,
            targets="label",
            predictions="predictions",
            model_type="classifier",
        )
        
        print('\nXGBoost Results:')
        print(f'Acurácia: {accuracy:.4f}')
        print(f'Precisão: {precision:.4f}')
        print(f'Revocação: {recall:.4f}')
        print(f'F1-score: {f1:.4f}')

# Executar os modelos
if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    train_random_forest(X_train, X_test, y_train, y_test)
    train_xgboost(X_train, X_test, y_train, y_test)