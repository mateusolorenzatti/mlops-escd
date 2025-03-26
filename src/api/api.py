from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

mlflow.set_tracking_uri("sqlite:///../../mlflow.db")

client = mlflow.MlflowClient()
models = client.search_registered_models()

model_names = [model.name for model in models]

loaded_models = {}

STAGE = "Production"

class PredictionInput(BaseModel):
    HighBP: int
    HighChol: int
    BMI: float
    Stroke: int
    HeartDiseaseorAttack: int
    PhysActivity: int
    HvyAlcoholConsump: int
    GenHlth: int
    MentHlth: int
    PhysHlth: int
    DiffWalk: int
    Age: int
    Income: int

for model_name in model_names:
    print(f"Carregando modelo {model_name} do MLflow...")

    loaded_models[model_name] = mlflow.pyfunc.load_model(f"models:/{model_name}/{STAGE}")

    print(f"Modelo {model_name} carregado com sucesso!")

@app.post("/predict/{model}")
def predict(model: str, data: PredictionInput):
    if model not in model_names:
        raise HTTPException(status_code=404, detail=f"Modelo '{model}' inexistente")

    df = pd.DataFrame([data.dict()])

    try:
        prediction = loaded_models[model].predict(df) 

        return {"model": model, "prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar previsão: {str(e)}")

@app.get("/")
def home():
    return {"message": f"API de Previsões usando MLFlow está rodando!"}

# uvicorn api:app --host 0.0.0.0 --port 8000 --reload

