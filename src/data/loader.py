import json
from pathlib import Path

import joblib
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def get_target(name: str) -> str:
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    return cfg["datasets"][name]["target"]


def load_dataset(name: str, split: str = "train") -> pd.DataFrame:
    """name: 'household' | 'appliances' ; split: 'train' | 'val' | 'test'"""
    return pd.read_parquet(PROCESSED / f"{name}_{split}.parquet")


def get_xy(df, target):
    return df.drop(columns=[target]), df[target]


def load_feature_list(name: str) -> list:
    return json.loads((MODELS / f"{name}_features.json").read_text(encoding="utf-8"))


def load_scaler(name: str):
    return joblib.load(MODELS / f"{name}_scaler.joblib")


def get_scaled_xy(name: str, split: str = "train"):
    """X normalisé (pour le LSTM) et y brut, colonnes dans l'ordre de feature_list."""
    df = load_dataset(name, split)
    feats = load_feature_list(name)
    X = pd.DataFrame(load_scaler(name).transform(df[feats]), index=df.index, columns=feats)
    return X, df[get_target(name)]