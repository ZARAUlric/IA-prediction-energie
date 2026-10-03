import pandas as pd
from pathlib import Path
from src.data.loader import load_dataset, load_feature_list
from src.monitoring.drift import detect_drift, save_drift_report


def evaluate_drift(dataset_name, reference_split="train", current_split="test"):
    """Évalue la dérive entre deux périodes"""
    
    print(f"=== Évaluation de dérive pour {dataset_name} ===")
    
    # Charger les données
    reference = load_dataset(dataset_name, reference_split)
    current = load_dataset(dataset_name, current_split)
    
    features = load_feature_list(dataset_name)
    
    # Détecter la dérive
    drift_report = detect_drift(reference, current, features)
    
    # Sauvegarder le rapport
    output_path = Path("monitoring") / f"{dataset_name}_drift_{reference_split}_vs_{current_split}.json"
    save_drift_report(drift_report, output_path)
    
    print(f"\nRapport de dérive:")
    print(f"Dérive globale détectée: {drift_report['overall_drift']}")
    
    for feature, metrics in drift_report["features"].items():
        if metrics["psi_drift"] or metrics["ks_drift"]:
            print(f"⚠️ {feature}: PSI={metrics['psi']:.3f}, KS p-value={metrics['ks_p_value']:.4f}")
    
    print(f"\n✅ Rapport sauvegardé dans {output_path}")
    
    return drift_report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Évaluer la dérive des données")
    parser.add_argument("--dataset", type=str, required=True, choices=["household", "appliances"])
    parser.add_argument("--reference", type=str, default="train", help="Split de référence")
    parser.add_argument("--current", type=str, default="test", help="Split actuel")
    
    args = parser.parse_args()
    
    evaluate_drift(args.dataset, args.reference, args.current)
