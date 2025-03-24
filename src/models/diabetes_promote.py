import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="sqlite:///../../mlflow.db")  # Ajustando o caminho correto

staging_threshold = 0.56 

models = ["RandomForest", "XGBoost"]

for model_name in models:
    versions = client.search_model_versions(f"name='{model_name}'")

    best_model = None
    best_f1_score = 0

    for version in versions:
        run_id = version.run_id
        metrics = client.get_run(run_id).data.metrics

        if "f1_score" in metrics:
            f1 = metrics["f1_score"]

            if f1 > staging_threshold:
                client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage="Staging"
                )
                print(f"Modelo {model_name} versão {version.version} com F1-score {f1} movido para Staging.")

            # Encontrar o melhor modelo para Produção
            if f1 > best_f1_score:
                best_f1_score = f1
                best_model = version.version

    # Atualizar o Champion (Produção)
    if best_model:
        client.transition_model_version_stage(
            name=model_name,
            version=best_model,
            stage="Production"
        )
        print(f"Modelo {model_name} versão {best_model} agora é o Champion com F1-score {best_f1_score}.")
    else:
        print(f"Nenhum modelo {model_name} atende ao critério para ser Champion.")
