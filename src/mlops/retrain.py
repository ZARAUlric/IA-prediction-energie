import argparse
from pathlib import Path
import json

from src.monitoring.drift import detect_drift, save_drift_report
from src.data.loader import load_dataset, load_feature_list
from src.models.train import train_model
from src.mlops.registry import register_model, promote_to_production


def check_and_retrain(dataset_name, model_type="xgboost", psi_threshold=0.2):
    """Vérifie la dérive et réentraîne si nécessaire"""
    
    print(f"=== Vérification de dérive pour {dataset_name} ===")
    
    # Charger les données
    train = load_dataset(dataset_name, "train")
    test = load_dataset(dataset_name, "test")
    
    features = load_feature_list(dataset_name)
    
    # Détecter la dérive
    drift_report = detect_drift(train, test, features, psi_threshold)
    
    # Sauvegarder le rapport
    output_path = Path("monitoring") / f"{dataset_name}_drift_check.json"
    save_drift_report(drift_report, output_path)
    
    if drift_report["overall_drift"]:
        print("⚠️ Dérive détectée ! Réentraînement nécessaire...")
        
        # Réentraîner le modèle
        model, metrics = train_model(model_type, dataset_name)
        
        # Enregistrer dans MLflow
        import mlflow
        run_id = mlflow.active_run().info.run_id
        
        # Enregistrer dans le registry
        model_name = f"{model_type}_{dataset_name}"
        register_model(model_name, run_id, stage="Staging")
        
        print(f"✅ Modèle réentraîné et enregistré: {model_name}")
        
        return {
            "drift_detected": True,
            "retrained": True,
            "model_name": model_name,
            "metrics": metrics
        }
    else:
        print("✅ Pas de dérive détectée. Réentraînement non nécessaire.")
        
        return {
            "drift_detected": False,
            "retrained": False
        }


def scheduled_retrain():
    """Réentraînement planifié (ex: hebdomadaire)"""
    datasets = ["household", "appliances"]
    model_type = "xgboost"
    
    results = {}
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Traitement de {dataset}")
        print('='*60)
        
        try:
            result = check_and_retrain(dataset, model_type)
            results[dataset] = result
        except Exception as e:
            print(f"❌ Erreur avec {dataset}: {e}")
            results[dataset] = {"error": str(e)}
    
    # Sauvegarder le résumé
    summary_path = Path("monitoring") / "retrain_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Résumé sauvegardé dans {summary_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Réentraînement automatique")
    parser.add_argument("--dataset", type=str, choices=["household", "appliances"],
                        help="Dataset spécifique (sinon tous)")
    parser.add_argument("--model", type=str, default="xgboost",
                        help="Type de modèle")
    parser.add_argument("--psi_threshold", type=float, default=0.2,
                        help="Seuil PSI pour détecter la dérive")
    
    args = parser.parse_args()
    
    if args.dataset:
        check_and_retrain(args.dataset, args.model, args.psi_threshold)
    else:
        scheduled_retrain()
