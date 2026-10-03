# MLOps et Monitoring

Ce dossier contient les outils MLOps pour le déploiement, le monitoring et le réentraînement automatique.

## 📁 Structure

```
src/mlops/
├── __init__.py
├── registry.py      # MLflow Model Registry
└── retrain.py       # Réentraînement automatique

src/monitoring/
├── __init__.py
├── drift.py         # Détection de dérive (PSI, KS test)
└── evaluate.py      # Script d'évaluation de dérive
```

## 🚀 Utilisation

### MLflow Model Registry

```powershell
# Enregistrer un modèle dans le registry
python -m src.mlops.registry --action register --model_name xgboost_household --run_id <run_id>

# Promouvoir un modèle en production
python -m src.mlops.registry --action promote --model_name xgboost_household --version 1

# Charger le modèle en production
python -m src.mlops.registry --action get --model_name xgboost_household
```

### Monitoring de dérive

```powershell
# Évaluer la dérive entre train et test
python -m src.monitoring.evaluate --dataset household --reference train --current test

# Évaluer la dérive sur appliances
python -m src.monitoring.evaluate --dataset appliances
```

### Réentraînement automatique

```powershell
# Vérifier la dérive et réentraîner si nécessaire
python -m src.mlops.retrain --dataset household --model xgboost

# Réentraîner tous les datasets
python -m src.mlops.retrain
```

## 📊 Métriques de dérive

### PSI (Population Stability Index)
- **< 0.1** : Pas de dérive significative
- **0.1 - 0.2** : Dérive légère
- **> 0.2** : Dérive significative (réentraînement recommandé)

### KS Test (Kolmogorov-Smirnov)
- **p-value < 0.05** : Dérive détectée (distributions différentes)

## 🐳 Docker

### Construire et lancer avec docker-compose

```powershell
# Construire les images
docker-compose build

# Lancer tous les services
docker-compose up -d

# Voir les logs
docker-compose logs -f api
docker-compose logs -f front
docker-compose logs -f mlflow
```

### Services

- **api** : FastAPI sur port 8000
- **front** : Gradio sur port 7860
- **mlflow** : MLflow UI sur port 5000

## 📡 API Endpoints

### Chargement de modèle
```bash
POST /model/load
{
  "model_type": "xgboost",
  "dataset": "household",
  "model_path": "models/xgboost_household.joblib"  # optionnel
}
```

### Prédiction
```bash
POST /predict
{
  "dataset": "household",
  "features": {
    "global_reactive_power_lag_1": 0.5,
    "voltage_lag_1": 230,
    ...
  }
}
```

### Informations modèle
```bash
GET /model/info
```

## 🔄 Pipeline CI/CD

Le pipeline de réentraînement peut être automatisé avec :

1. **GitHub Actions** : Lancer le réentraînement chaque semaine
2. **Cron job** : Planifier l'exécution de `src/mlops/retrain.py`
3. **Prefect** : Intégrer dans le pipeline existant

Exemple de cron (hebdomadaire, dimanche 2h) :
```
0 2 * * 0 cd /path/to/project && python -m src.mlops.retrain >> /var/log/energy_retrain.log 2>&1
```

## 📝 Rapports

Les rapports de dérive sont sauvegardés dans `monitoring/` :
- `{dataset}_drift_train_vs_test.json` : Rapport de dérive
- `retrain_summary.json` : Résumé du réentraînement

## ⚠️ Notes

- Le réentraînement automatique nécessite que MLflow soit configuré
- Les modèles doivent être sauvegardés avec joblib pour être chargés dans l'API
- Le seuil PSI par défaut est 0.2 (ajustable via `--psi_threshold`)
