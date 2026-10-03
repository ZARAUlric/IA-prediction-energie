import joblib
from pathlib import Path

from src.data.loader import load_dataset, get_xy, get_target
from src.models.xgboost_model import XGBoostModel


def save_best_model(dataset_name="household", model_type="xgboost"):
    """Entraîne et sauvegarde le meilleur modèle"""
    
    print(f"=== Entraînement et sauvegarde du meilleur modèle ({model_type}) ===")
    
    # Charger les données
    train = load_dataset(dataset_name, "train")
    val = load_dataset(dataset_name, "val")
    
    target = get_target(dataset_name)
    X_train, y_train = get_xy(train, target)
    X_val, y_val = get_xy(val, target)
    
    # Combiner train + val pour l'entraînement final
    X_full = pd.concat([X_train, X_val])
    y_full = pd.concat([y_train, y_val])
    
    print(f"Entraînement sur {len(X_full)} lignes")
    
    # Créer et entraîner le modèle
    model = XGBoostModel()
    model.fit(X_full, y_full)
    
    # Sauvegarder
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / f"{model_type}_{dataset_name}.joblib"
    joblib.dump(model.model, model_path)
    
    print(f"✅ Modèle sauvegardé dans {model_path}")
    
    return model_path


if __name__ == "__main__":
    import pandas as pd
    
    save_best_model("household", "xgboost")
