import mlflow
import mlflow.sklearn
import mlflow.pytorch
import json
from pathlib import Path


def setup_mlflow(experiment_name="energy_prediction"):
    """Configure MLflow pour le tracking des expériences"""
    mlflow.set_experiment(experiment_name)
    # Retourner l'expérience créée/récupérée
    return mlflow.get_experiment_by_name(experiment_name)


def log_model(model, model_name, model_type="sklearn"):
    """Log un modèle dans MLflow"""
    if model_type == "sklearn":
        mlflow.sklearn.log_model(model, model_name)
    elif model_type == "pytorch":
        mlflow.pytorch.log_model(model, model_name)
    else:
        mlflow.log_artifact(model, model_name)


def log_params(params):
    """Log les paramètres d'entraînement"""
    mlflow.log_params(params)


def log_metrics(metrics):
    """Log les métriques d'évaluation"""
    mlflow.log_metrics(metrics)


def log_feature_importance(importance_dict, artifact_name="feature_importance.json"):
    """Log l'importance des features"""
    with open(artifact_name, 'w') as f:
        json.dump(importance_dict, f, indent=2)
    mlflow.log_artifact(artifact_name)


def log_artifact(file_path, artifact_path=None):
    """Log un fichier artefact"""
    mlflow.log_artifact(file_path, artifact_path)


def get_best_model(metric="mae", experiment_name="energy_prediction"):
    """Récupère le meilleur modèle selon une métrique"""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if len(runs) == 0:
        return None
    
    best_run = runs.sort_values(f"metrics.{metric}", ascending=True).iloc[0]
    return best_run
