# Prédiction de la consommation énergétique

Projet d'école (équipe de 4) : prédiction de la consommation d'énergie avec XGBoost, Random Forest, LSTM et Prophet, avec une chaîne MLOps (pipeline, versionnage des modèles, API, monitoring).

## Structure

```
configs/config.yaml     paramètres du pipeline (fréquence, lags, découpage...)
data/raw/               données brutes (non versionnées)
data/processed/         jeux prêts pour les modèles (non versionnés)
docs/data schema.md     description des colonnes et règles de construction
models/                 scalers et listes de features
notebooks/              exploration
src/data/               téléchargement, nettoyage, loader
src/features/           features et découpage temporel
src/pipeline/           pipeline Prefect, simulation, données de dérive
src/models/             interface commune des modèles
tests/                  tests pytest
```

## Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduire les données

```powershell
python -m src.data.download              # télécharge les 2 datasets dans data/raw/
python -m src.pipeline.run_pipeline      # nettoie, crée les features, découpe, sauvegarde
python -m pytest -v                      # vérifie les données (16 tests)
```

Le pipeline produit dans `data/processed/` les fichiers `<dataset>_{train,val,test}.parquet` et dans `models/` le scaler et la liste des features.

## Utiliser les données

```python
from src.data.loader import load_dataset, get_xy, get_target, get_scaled_xy

df = load_dataset("household", "train")          # "household" ou "appliances"
X, y = get_xy(df, get_target("household"))       # arbres, Prophet
X_s, y = get_scaled_xy("household", "train")     # LSTM (features normalisées)
```

## Entraînement périodique et simulation

```powershell
python -m src.pipeline.schedule                       # exécution planifiée (dimanche 2h)
python -m src.pipeline.simulate_runs household 4      # 4 runs sur des données croissantes
python -m src.pipeline.make_drift_data mean_shift     # jeu de données avec dérive (voir docs)
```

## Règles à respecter

- Split chronologique uniquement, jamais aléatoire.
- Utiliser `load_dataset` : ne pas relire les fichiers bruts.
- Même ordre de features partout (`models/*_features.json`).
- Toute modification du schéma est annoncée à l'équipe et notée dans `docs/data schema.md`.

Message pour l'équipe

Salut ! Le bloc 1 (données + pipeline) est livré.

Branche : feature/bloc1-data-pipeline (PR ouverte)
Données : energie_data_v1.zip sur le drive -> parquet dans data/processed/,
          contenu de models/ dans models/
Doc : docs/data schema.md (colonnes, règles de nettoyage, dates de découpage)
README : installation + commandes

Chargement en une ligne :
  from src.data.loader import load_dataset, get_xy, get_target
  df = load_dataset("household", "train")   # ou "appliances", split train/val/test

À savoir :
- Toutes les mesures capteurs/météo sont décalées d'1h (_lag_1) pour éviter la fuite
  de données : à l'heure t on n'utilise que des infos <= t-1.
- LSTM (P3) : l'index a quelques trous, ne pas faire de fenêtres qui les enjambent.
- Pour comparer les modèles, tous sur les mêmes jeux val/test de data/processed/
  (pas ceux de runs/).
- Appliances est court (4,5 mois, ~2 200 lignes de train) : limite à citer dans le rapport.
- P4 : je vous prépare des jeux de données avec dérive pour tester le monitoring.
- P2/P3 : l'entraînement se branche dans run_pipeline.py via un paramètre train_fn ;
  dites-moi quand vos fonctions train(...) sont prêtes.

Si une colonne ou un format vous gêne, dites-le-moi avant qu'on fige le schéma.
