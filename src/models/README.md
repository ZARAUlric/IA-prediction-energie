# Modèles de prédiction

Ce dossier contient tous les modèles pour la prédiction de consommation énergétique.

## 📁 Structure

```
src/models/
├── __init__.py
├── metrics.py              # Métriques d'évaluation (MAE, RMSE, MAPE)
├── baselines.py            # Baselines (moyenne, persistance, rolling mean)
├── xgboost_model.py        # Modèle XGBoost
├── random_forest_model.py   # Modèle Random Forest
├── prophet_model.py        # Modèle Prophet
├── lstm_model.py           # Modèle LSTM (PyTorch)
├── mlflow_utils.py         # Utilitaires MLflow
├── train.py                # Script d'entraînement unifié
└── compare.py              # Script de comparaison des modèles
```

## 🚀 Installation des dépendances

```powershell
pip install prophet==1.1.5 torch==2.6.0 mlflow==2.21.0 xgboost scikit-learn
```

## 📖 Utilisation

### Entraîner un modèle

```powershell
# Baseline
python -m src.models.train --model mean --dataset household
python -m src.models.train --model persistence --dataset household

# Modèles classiques
python -m src.models.train --model xgboost --dataset household
python -m src.models.train --model random_forest --dataset household
python -m src.models.train --model prophet --dataset household

# LSTM
python -m src.models.train --model lstm --dataset household
```

### Comparer plusieurs modèles

```powershell
# Comparer tous les modèles (sauf LSTM)
python -m src.models.compare --dataset household

# Comparer des modèles spécifiques
python -m src.models.compare --dataset household --models mean persistence xgboost random_forest
```

### MLflow Tracking

Les expériences sont automatiquement trackées avec MLflow. Pour visualiser :

```powershell
mlflow ui
```

Puis ouvrir `http://localhost:5000`

## 📊 Modèles disponibles

### Baselines
- **mean** : Prédit la moyenne de la cible sur le train
- **persistence** : Prédit la dernière valeur connue (lag 1)
- **rolling_mean** : Prédit la moyenne glissante sur 24h

### Modèles classiques
- **xgboost** : Gradient Boosting avec hyperparamètres
- **random_forest** : Random Forest avec hyperparamètres
- **prophet** : Modèle Facebook Prophet avec saisonnalités

### Deep Learning
- **lstm** : Réseau LSTM avec PyTorch (early stopping)

## 📈 Métriques

- **MAE** (Mean Absolute Error) : Erreur absolue moyenne
- **RMSE** (Root Mean Squared Error) : Racine de l'erreur quadratique moyenne
- **MAPE** (Mean Absolute Percentage Error) : Erreur absolue moyenne en pourcentage

## 🔧 Hyperparamètres

### XGBoost
- n_estimators: [50, 100, 200]
- max_depth: [3, 6, 9]
- learning_rate: [0.01, 0.1, 0.2]
- subsample: [0.6, 0.8, 1.0]

### Random Forest
- n_estimators: [50, 100, 200]
- max_depth: [5, 10, 15, None]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]

### LSTM
- hidden_size: 64
- num_layers: 2
- dropout: 0.2
- sequence_length: 24
- learning_rate: 0.001
- epochs: 50
- batch_size: 32
- early_stopping_patience: 10

## 📝 Résultats

Les résultats de comparaison sont sauvegardés dans `results/<dataset>_comparison.csv`

## ⚠️ Notes

- Le LSTM nécessite des données normalisées (via `get_scaled_xy`)
- Prophet nécessite un index datetime
- Les baselines servent de référence pour évaluer la performance des modèles avancés
