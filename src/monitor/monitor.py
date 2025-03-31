import pandas as pd
import numpy as np
import requests
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from sklearn.preprocessing import LabelEncoder
import os
import json

def check_for_drift(drift_score, drift_by_columns):
    num_columns_drift = sum(1 for col, values in drift_by_columns.items() if values.get("drift_detected", False))
    if drift_score > 0.5:
        print("Drift detectado no Dataset")
        os.system("python3 ../models/diabetes.py")
    elif num_columns_drift > 2:
        print(f"Drift detectado em {num_columns_drift} colunas! Treinando novo modelo...")
        os.system("python3 ../models/diabetes.py")
    else:
        print("Modelo ainda está bom, sem necessidade de re-treinamento.")

def load_new_data():
    df = pd.read_csv("../../data/diabetes_binary.csv")
    df = df.sample(n=1000, random_state=42)
    X, y = preprocess_data(df)
    return X, y

def simulate_drift(df_examples):
    new_data = df_examples.copy()
    new_data["BMI"] = np.random.uniform(15, 40, new_data.shape[0])
    new_data["Income"] *= 1.2
    print("Criado dataset artificialmente alterado para simular drift.")
    return new_data

def preprocess_data(df):
    df.replace({"Yes": 1, "No": 0}, inplace=True)
    df = df.infer_objects(copy=False)
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("float64")
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    df.fillna(0, inplace=True)
    X = df.drop(columns=["Diabetes_binary"], errors="ignore")
    y = df["Diabetes_binary"]
    return X, y.astype(int)

def get_predictions(data):
    api_url = "http://0.0.0.0:8000/predict/RandomForest"
    headers = {"Content-Type": "application/json"}
    predictions = []
    
    for _, row in data.iterrows():
        instance = row.to_dict()
        response = requests.post(api_url, headers=headers, json=instance)
        if response.status_code == 200:
            try:
                pred = response.json().get("prediction", None)
                predictions.append(pred if pred is not None else -1)
            except json.JSONDecodeError:
                predictions.append(-1)
        else:
            predictions.append(-1)
    
    return predictions

def evaluate_model(df, y, new_data):
    if new_data is None:
        print("Avaliando modelo com dados originais")
        df["prediction"] = get_predictions(df)
    else:
        print("Avaliando modelo com dados artificiais")
        new_data["prediction"] = get_predictions(new_data)
    
    df["target"] = y
    report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
    report.run(reference_data=df, current_data=new_data if new_data is not None else df)
    report.save_html("monitoring_report.html")
    report_dict = report.as_dict()
    drift_score = report_dict["metrics"][0]["result"]["dataset_drift"]
    drift_by_columns = report_dict["metrics"][1]["result"].get("drift_by_columns", {})
    return drift_score, drift_by_columns

def main():
    df_examples, y = load_new_data()
    drift_score, drift_by_columns = evaluate_model(df_examples, y, None)
    new_data = simulate_drift(df_examples)
    drift_score, drift_by_columns = evaluate_model(df_examples, y, new_data)
    check_for_drift(drift_score, drift_by_columns)

if __name__ == "__main__":
    main()
