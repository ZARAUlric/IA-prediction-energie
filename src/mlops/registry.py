import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
import joblib


def setup_registry(experiment_name="energy_prediction"):
    """Configure le MLflow Model Registry"""
    client = MlflowClient()
    
    # Créer ou récupérer l'expérience
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    print(f"Expérience ID: {experiment_id}")
    return client, experiment_id


def register_model(model_name, run_id, stage="Staging"):
    """Enregistre un modèle dans le Model Registry"""
    client = MlflowClient()
    
    # Enregistrer le modèle
    model_uri = f"runs:/{run_id}/model"
    model_version = mlflow.register_model(model_uri, model_name)
    
    # Transitions de stage
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage=stage
    )
    
    print(f"Modèle {model_name} version {model_version.version} enregistré en {stage}")
    return model_version


def get_production_model(model_name):
    """Récupère le modèle en production"""
    client = MlflowClient()
    
    # Récupérer la version en production
    model_version_info = client.get_latest_versions(model_name, stages=["Production"])
    
    if not model_version_info:
        return None
    
    version = model_version_info[0].version
    model_uri = f"models:/{model_name}/{version}"
    
    # Charger le modèle
    model = mlflow.sklearn.load_model(model_uri)
    
    print(f"Modèle {model_name} version {version} chargé depuis Production")
    return model


def promote_to_production(model_name, version):
    """Promeut un modèle en production"""
    client = MlflowClient()
    
    # Archiver l'ancien modèle en production
    current_prod = client.get_latest_versions(model_name, stages=["Production"])
    if current_prod:
        client.transition_model_version_stage(
            name=model_name,
            version=current_prod[0].version,
            stage="Archived"
        )
    
    # Promouvoir le nouveau
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production"
    )
    
    print(f"Modèle {model_name} version {version} promu en Production")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gérer le MLflow Model Registry")
    parser.add_argument("--action", type=str, required=True, choices=["register", "promote", "get"])
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--run_id", type=str, help="Run ID pour l'enregistrement")
    parser.add_argument("--version", type=str, help="Version pour la promotion")
    
    args = parser.parse_args()
    
    if args.action == "register":
        if not args.run_id:
            print("--run_id requis pour l'enregistrement")
        else:
            register_model(args.model_name, args.run_id)
    
    elif args.action == "promote":
        if not args.version:
            print("--version requis pour la promotion")
        else:
            promote_to_production(args.model_name, args.version)
    
    elif args.action == "get":
        model = get_production_model(args.model_name)
        if model:
            print("Modèle chargé avec succès")
        else:
            print("Aucun modèle en production")
