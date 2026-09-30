import json
from pathlib import Path

import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


def temporal_split(df, train_end=0.70, val_end=0.85):
    n = len(df)
    i1, i2 = int(n * train_end), int(n * val_end)
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def cv_splits(n_splits=5):
    """Pour la validation croisée temporelle (Personne 2)."""
    return TimeSeriesSplit(n_splits=n_splits)


def fit_scaler(train, feature_cols):
    """Ajusté sur le TRAIN uniquement."""
    return StandardScaler().fit(train[feature_cols])


def save_splits(train, val, test, name, out="data/processed"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out / f"{name}_train.parquet")
    val.to_parquet(out / f"{name}_val.parquet")
    test.to_parquet(out / f"{name}_test.parquet")

    info = {
        split: {
            "start": str(d.index.min()),
            "end": str(d.index.max()),
            "rows": len(d),
        }
        for split, d in [("train", train), ("val", val), ("test", test)]
    }
    (out / f"{name}_split_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def save_artifacts(name, scaler, feature_list, out="models"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, out / f"{name}_scaler.joblib")
    (out / f"{name}_features.json").write_text(
        json.dumps(feature_list, indent=2), encoding="utf-8"
    )