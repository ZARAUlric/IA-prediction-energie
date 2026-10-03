import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import json


def calculate_psi(expected, actual, bins=10):
    """Population Stability Index (PSI) pour détecter la dérive"""
    # Créer les bins basés sur expected
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    
    # Calculer les distributions
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)
    
    # Convertir en proportions
    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)
    
    # Éviter la division par zéro
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    # Calculer PSI
    psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    
    return psi


def ks_test(expected, actual):
    """Kolmogorov-Smirnov test pour détecter la dérive"""
    statistic, p_value = stats.ks_2samp(expected, actual)
    return {
        "statistic": statistic,
        "p_value": p_value,
        "drift_detected": p_value < 0.05
    }


def detect_drift(reference_df, current_df, features, psi_threshold=0.2):
    """Détecte la dérive sur plusieurs features"""
    drift_report = {
        "features": {},
        "overall_drift": False
    }
    
    for feature in features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue
        
        expected = reference_df[feature].dropna()
        actual = current_df[feature].dropna()
        
        if len(expected) == 0 or len(actual) == 0:
            continue
        
        # PSI
        psi = calculate_psi(expected, actual)
        
        # KS test
        ks_result = ks_test(expected, actual)
        
        drift_report["features"][feature] = {
            "psi": psi,
            "psi_drift": psi > psi_threshold,
            "ks_statistic": ks_result["statistic"],
            "ks_p_value": ks_result["p_value"],
            "ks_drift": ks_result["drift_detected"]
        }
        
        # Marquer si dérive détectée
        if psi > psi_threshold or ks_result["drift_detected"]:
            drift_report["overall_drift"] = True
    
    return drift_report


def save_drift_report(report, output_path="monitoring/drift_report.json"):
    """Sauvegarde le rapport de dérive"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convertir les booléens et numpy types pour JSON
    def convert_types(obj):
        if isinstance(obj, (bool, np.bool_)):
            return int(obj)
        elif isinstance(obj, (int, np.integer)):
            return int(obj)
        elif isinstance(obj, (float, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    report_serializable = convert_types(report)
    
    with open(output_path, 'w') as f:
        json.dump(report_serializable, f, indent=2)
    
    return output_path


def check_prediction_drift(y_true, y_pred, mae_threshold_factor=2.0):
    """Détecte la dérive dans les prédictions (MAE glissant)"""
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Seuil simple : si MAE > 2x la médiane historique (à adapter)
    # Pour l'instant, retourne juste le MAE
    return {
        "mae": mae,
        "n_samples": len(y_true),
        "drift_detected": False  # À implémenter avec historique
    }
