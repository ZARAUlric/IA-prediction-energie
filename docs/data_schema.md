# Schéma des données : prédiction de la consommation énergétique

Document du bloc 1 (données et pipeline). Il décrit les fichiers livrés aux autres blocs.

## 1. Vue d'ensemble

| | Household | Appliances |
|---|---|---|
| Source | UCI, Individual Household Electric Power Consumption | UCI, Appliances Energy Prediction |
| Période brute | 2006-12-16 → 2010-11-26 | 2016-01-11 → 2016-05-27 |
| Pas de temps brut | 1 minute (2 075 259 lignes) | 10 minutes (19 735 lignes) |
| Pas de temps livré | **1 heure** | **1 heure** |
| Variable cible | `global_active_power` (kW moyen sur l'heure) | `appliances` (Wh cumulés sur l'heure) |
| Colonnes livrées | 31 (1 cible + 30 features) | 49 (1 cible + 48 features) |
| Lignes train / val / test | 23 270 / 4 986 / 4 987 | 2 184 / 468 / 469 |
| Jours fériés | France (`FR`) | Belgique (`BE`) |

Dates exactes de chaque découpage : voir `data/processed/<dataset>_split_info.json`.

## 2. Fichiers livrés

```
data/processed/
  household_train.parquet   household_val.parquet   household_test.parquet
  appliances_train.parquet  appliances_val.parquet  appliances_test.parquet
  household_split_info.json appliances_split_info.json
models/
  household_features.json   household_scaler.joblib
  appliances_features.json  appliances_scaler.joblib
```

- Index de chaque parquet : `DatetimeIndex` horaire, trié, sans doublon.
- La **cible est la première colonne** ; toutes les autres colonnes sont des features.
- `*_features.json` : liste ordonnée des features (à respecter dans le même ordre à l'entraînement et dans l'API).
- `*_scaler.joblib` : `StandardScaler` ajusté **sur le train uniquement**.

## 3. Utilisation

```python
from src.data.loader import load_dataset, get_xy, get_target, get_scaled_xy

df = load_dataset("household", "train")             # DataFrame complet
X, y = get_xy(df, get_target("household"))          # X brut (arbres, Prophet)
X_s, y = get_scaled_xy("household", "train")        # X normalisé (LSTM)
```

## 4. Règles de construction

### Nettoyage
1. Valeurs manquantes (`?` dans Household) converties en NaN ; doublons d'index supprimés.
2. Valeurs impossibles (puissance < 0, tension ≤ 0) converties en NaN.
3. Rééchantillonnage à l'heure. Une heure est invalide (NaN) si moins de 50 % de ses mesures sont valides.
4. Interpolation temporelle **uniquement des trous ≤ 3 heures**. Les trous plus longs restent vides, puis les lignes concernées sont supprimées.
5. Les pics de consommation ne sont **pas supprimés** (ce sont de vraies consommations).
6. Appliances : colonnes de bruit `rv1` et `rv2` supprimées.

### Absence de fuite de données
- Toutes les mesures de la même heure que la cible (capteurs, météo, tension, intensité, etc.) sont **décalées d'une heure** (`_lag_1`).
- Les moyennes et écarts-types glissants sont calculés sur `cible.shift(1)` : ils n'incluent jamais la valeur courante.
- Le scaler est ajusté sur le train uniquement.

### Découpage
- Chronologique, jamais aléatoire : 70 % train, 15 % validation, 15 % test.
- Pour la validation croisée : `src.features.split.cv_splits(n_splits)` (`TimeSeriesSplit`).

## 5. Colonnes : Household

| Colonne | Unité | Description |
|---|---|---|
| `global_active_power` | kW | **Cible.** Puissance active moyenne sur l'heure (= kWh consommés sur l'heure) |
| `global_reactive_power_lag_1` | kVAR | Puissance réactive moyenne de l'heure précédente |
| `voltage_lag_1` | V | Tension moyenne de l'heure précédente |
| `global_intensity_lag_1` | A | Intensité moyenne de l'heure précédente |
| `sub_metering_1_lag_1` | Wh | Énergie du compteur 1 (cuisine) de l'heure précédente |
| `sub_metering_2_lag_1` | Wh | Énergie du compteur 2 (buanderie) de l'heure précédente |
| `sub_metering_3_lag_1` | Wh | Énergie du compteur 3 (chauffe-eau, climatisation) de l'heure précédente |
| `sub_metering_rest_lag_1` | Wh | Énergie non mesurée par les 3 compteurs, heure précédente |
| `global_active_power_lag_{1,2,3,24,48,168}` | kW | Cible décalée de 1, 2, 3 h, 1 jour, 2 jours, 1 semaine |
| `global_active_power_rollmean_{3,6,24}` | kW | Moyenne glissante sur 3, 6, 24 h (fenêtre finissant à t-1) |
| `global_active_power_rollstd_{3,6,24}` | kW | Écart-type glissant sur 3, 6, 24 h (fenêtre finissant à t-1) |
| `hour` | 0-23 | Heure de la journée |
| `dayofweek` | 0-6 | Jour de la semaine (0 = lundi) |
| `month` | 1-12 | Mois |
| `is_weekend` | 0/1 | Samedi ou dimanche |
| `is_holiday` | 0/1 | Jour férié |
| `hour_sin`, `hour_cos` | -1 à 1 | Encodage cyclique de