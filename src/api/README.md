# API FastAPI

API REST pour le projet de prédiction de consommation énergétique.

## 🚀 Installation

Les dépendances sont déjà dans `requirements.txt` :
- fastapi>=0.142.2
- uvicorn>=0.54.0
- pydantic>=2.13.5

## 📖 Utilisation

### Lancer l'API

```powershell
# Depuis la racine du projet
python -m src.api.main
```

Ou directement :
```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

L'API sera accessible sur `http://localhost:8000`

### Documentation interactive

Ouvrez `http://localhost:8000/docs` dans votre navigateur pour la documentation Swagger UI.

## 📡 Endpoints

### GET `/`
Racine de l'API avec liste des endpoints.

### GET `/health`
Vérifie que l'API fonctionne.

**Réponse :**
```json
{
  "status": "healthy"
}
```

### GET `/datasets`
Liste les datasets disponibles.

**Réponse :**
```json
{
  "datasets": ["household", "appliances"]
}
```

### GET `/datasets/{name}/info`
Informations détaillées sur un dataset.

**Paramètres :**
- `name` : "household" ou "appliances"

**Réponse :**
```json
{
  "name": "household",
  "target": "global_active_power",
  "n_features": 30,
  "train_rows": 23270,
  "val_rows": 4986,
  "test_rows": 4987,
  "train_start": "2006-12-16 00:00:00",
  "train_end": "2010-01-22 23:00:00",
  "test_start": "2010-08-04 00:00:00",
  "test_end": "2010-11-26 23:00:00"
}
```

### POST `/data/sample`
Récupère un échantillon de données.

**Body :**
```json
{
  "dataset": "household",
  "split": "train",
  "n_rows": 10
}
```

**Réponse :**
```json
{
  "index": ["2006-12-16 01:00:00", ...],
  "columns": ["global_active_power", "global_reactive_power_lag_1", ...],
  "data": [[...], [...]],
  "shape": [10, 31]
}
```

### GET `/datasets/{name}/features`
Liste des features d'un dataset.

**Paramètres :**
- `name` : "household" ou "appliances"

**Réponse :**
```json
{
  "dataset": "household",
  "target": "global_active_power",
  "n_features": 30,
  "features": ["global_reactive_power_lag_1", "voltage_lag_1", ...]
}
```

## 🔧 Configuration

L'API utilise les fichiers de configuration du projet :
- `configs/config.yaml` : paramètres des datasets
- `data/processed/` : données traitées
- `models/` : scalers et listes de features

## 🧪 Tests

```powershell
# Tester l'API avec curl
curl http://localhost:8000/health
curl http://localhost:8000/datasets
curl http://localhost:8000/datasets/household/info
```

## 📝 Note

L'API nécessite que les données soient générées via le pipeline :
```powershell
python -m src.pipeline.run_pipeline
```
