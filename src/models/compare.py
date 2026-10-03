import argparse
import pandas as pd
from pathlib import Path

from src.data.loader import load_dataset, get_xy, get_target, get_scaled_xy
from src.models.metrics import evaluate

# Import des modèles avec Prophet optionnel
from src.models.baselines import MeanBaseline, PersistenceBaseline, RollingMeanBaseline
from src.models.xgboost_model import XGBoostModel
from src.models.random_forest_model import RandomForestModel
from src.models.lstm_model import LSTMRegressor

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

if PROPHET_AVAILABLE:
    MODELS["prophet"] = ProphetModel


def compare_models(dataset_name, models_to_compare=None):
    """Compare plusieurs modèles sur un dataset"""
    
    if models_to_compare is None:
        models_to_compare = ["mean", "persistence", "rolling_mean", "xgboost", "random_forest"]
        if PROPHET_AVAILABLE:
            models_to_compare.append("prophet")
    
    print(f"=== Comparaison des modèles sur {dataset_name} ===")
    
    results = []
    
    for model_name in models_to_compare:
        print(f"\n--- {model_name} ---")
        
        try:
            # Charger les données
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
                model = MODELS[model_name](input_size=input_size)
            else:
                X_train, y_train = get_xy(train, target)
                X_val, y_val = get_xy(val, target)
                X_test, y_test = get_xy(test, target)
                model = MODELS[model_name]()
            
            # Entraînement
            if model_name == "lstm":
                model.fit(X_train, y_train, X_val, y_val)
            else:
                model.fit(X_train, y_train)
            
            # Prédictions
            y_test_pred = model.predict(X_test)
            
            # Évaluation
            metrics = evaluate(y_test, y_test_pred)
            
            results.append({
                "model": model_name,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"]
            })
            
            print(f"MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, MAPE: {metrics['mape']:.2f}%")
            
        except Exception as e:
            print(f"❌ Erreur avec {model_name}: {e}")
            continue
    
    # Créer le tableau de résultats
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("mae")
    
    print("\n" + "="*60)
    print("TABLEAU COMPARATIF (trié par MAE)")
    print("="*60)
    print(df_results.to_string(index=False))
    
    # Sauvegarder les résultats
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{dataset_name}_comparison.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\n✅ Résultats sauvegardés dans {output_path}")
    
    return df_results


def main():
    parser = argparse.ArgumentParser(description="Comparer plusieurs modèles")
    parser.add_argument("--dataset", type=str, required=True, choices=["household", "appliances"],
                        help="Dataset à utiliser")
    parser.add_argument("--models", type=str, nargs="+", 
                        choices=list(MODELS.keys()),
                        help="Modèles à comparer (défaut: tous sauf LSTM)")
    
    args = parser.parse_args()
    
    compare_models(args.dataset, args.models)


if __name__ == "__main__":
    main()
