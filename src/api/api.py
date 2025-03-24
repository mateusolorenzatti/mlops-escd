from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

# 📌 Parâmetros para carregar o modelo do MLflow
MODEL_NAME = "RandomForest"  # Ou "XGBoost"
STAGE = "Production"

print(f"🔍 Carregando modelo {MODEL_NAME} do MLflow...")
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{STAGE}")
print(f"✅ Modelo {MODEL_NAME} carregado com sucesso!")

@app.post("/predict/")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"model": MODEL_NAME, "prediction": int(prediction[0])}

@app.get("/")
def home():
    return {"message": f"API de Previsões usando {MODEL_NAME} está rodando!"}

# uvicorn api:app --host 0.0.0.0 --port 8000 --reload

