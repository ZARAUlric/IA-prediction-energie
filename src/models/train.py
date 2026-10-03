import argparse
import sys
from pathlib import Path

import pandas as pd

from src.data.loader import load_dataset, get_xy, get_target, get_scaled_xy, load_feature_list
from src.models.metrics import evaluate
from src.models.mlflow_utils import setup_mlflow, log_params, log_metrics, log_model, log_feature_importance

# Modèles
from src.models.baselines import MeanBaseline, PersistenceBaseline, RollingMeanBaseline
from src.models.xgboost_model import XGBoostModel
from src.models.random_forest_model import RandomForestModel
from src.models.lstm_model import LSTMRegressor

# Prophet import optionnel (problème de compatibilité NumPy 2.x)
try:
    from src.models.prophet_model import ProphetModel
    PROPHET_AVAILABLE = True
except (ImportError, AttributeError):
    PROPHET_AVAILABLE = False
    ProphetModel = None


MODELS = {
    "mean": MeanBaseline,
    "persistence": PersistenceBaseline,
    "rolling_mean": RollingMeanBaseline,
    "xgboost": XGBoostModel,
    "random_forest": RandomForestModel,
    "lstm": LSTMRegressor,
}

# Ajouter Prophet seulement si disponible
if PROPHET_AVAILABLE:
    MODELS["prophet"] = ProphetModel


def train_model(model_name, dataset_name, tune=False):
    """Entraîne un modèle avec MLflow tracking"""
    
    # Setup MLflow
    setup_mlflow(f"energy_prediction_{dataset_name}")
    
    with mlflow.start_run(run_name=f"{model_name}_{dataset_name}"):
        print(f"=== Entraînement {model_name} sur {dataset_name} ===")
        
        # Charger les données
        print("Chargement des données...")
        train = load_dataset(dataset_name, "train")
        val = load_dataset(dataset_name, "val")
        test = load_dataset(dataset_name, "test")
        
        target = get_target(dataset_name)
        
        # Préparer X, y selon le modèle
        if model_name == "lstm":
            X_train, y_train = get_scaled_xy(dataset_name, "train")
            X_val, y_val = get_scaled_xy(dataset_name, "val")
            X_test, y_test = get_scaled_xy(dataset_name, "test")
            input_size = X_train.shape[1]
        else:
            X_train, y_train = get_xy(train, target)
            X_val, y_val = get_xy(val, target)
            X_test, y_test = get_xy(test, target)
            input_size = None
        
        print(f"Train: {len(X_train)} lignes, Val: {len(X_val)} lignes, Test: {len(X_test)} lignes")
        
        # Créer le modèle
        print(f"Création du modèle {model_name}...")
        if model_name == "lstm":
            model = MODELS[model_name](input_size=input_size)
        else:
            model = MODELS[model_name]()
        
        # Log des paramètres
        params = {
            "model": model_name,
            "dataset": dataset_name,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "n_features": X_train.shape[1]
        }
        if hasattr(model, 'params'):
            params.update(model.params)
        log_params(params)
        
        # Entraînement
        print("Entraînement...")
        if model_name == "lstm":
            model.fit(X_train, y_train, X_val, y_val)
        else:
            model.fit(X_train, y_train)
        
        # Prédictions
        print("Prédictions...")
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        # Évaluation
        print("Évaluation...")
        train_metrics = evaluate(y_train, y_train_pred)
        val_metrics = evaluate(y_val, y_val_pred)
        test_metrics = evaluate(y_test, y_test_pred)
        
        # Log des métriques
        for split, metrics in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
            log_metrics({f"{split}_{k}": v for k, v in metrics.items()})
        
        print(f"\n=== Résultats ===")
        print(f"Train MAE: {train_metrics['mae']:.4f}")
        print(f"Val MAE: {val_metrics['mae']:.4f}")
        print(f"Test MAE: {test_metrics['mae']:.4f}")
        print(f"Test RMSE: {test_metrics['rmse']:.4f}")
        print(f"Test MAPE: {test_metrics['mape']:.2f}%")
        
        # Log du modèle
        if model_name in ["xgboost", "random_forest"]:
            log_model(model.model, model_name, model_type="sklearn")
            # Log feature importance
            if hasattr(model, 'get_feature_importance'):
                importance = model.get_feature_importance()
                log_feature_importance(importance, f"{model_name}_feature_importance.json")
        elif model_name == "lstm":
            log_model(model.model, model_name, model_type="pytorch")
        else:
            log_model(model, model_name, model_type="sklearn")
        
        print(f"\n✅ Modèle {model_name} entraîné et loggé dans MLflow")
        
        return model, test_metrics


def main():
    parser = argparse.ArgumentParser(description="Entraîner un modèle de prédiction énergétique")
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()),
                        help="Modèle à entraîner")
    parser.add_argument("--dataset", type=str, required=True, choices=["household", "appliances"],
                        help="Dataset à utiliser")
    parser.add_argument("--tune", action="store_true", help="Lancer le tuning d'hyperparamètres")
    
    args = parser.parse_args()
    
    # Import mlflow ici pour éviter les imports circulaires
    import mlflow
    
    train_model(args.model, args.dataset, args.tune)


if __name__ == "__main__":
    main()
