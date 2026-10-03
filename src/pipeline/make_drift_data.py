import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.clean import load_household, clean_household
from src.data.loader import load_feature_list
from src.features.build import build_features

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ["mean_shift", "noise", "sensor_fault"]


def apply_drift(df, scenario, start, seed=42):
    df = df.copy()
    after = df.index >= pd.Timestamp(start)
    rng = np.random.default_rng(seed)

    if scenario == "mean_shift":
        # Consommation +30 % (nouvel équipement, changement d'usage...)
        cols = ["global_active_power", "global_intensity", "sub_metering_1",
                "sub_metering_2", "sub_metering_3", "sub_metering_rest"]
        df.loc[after, cols] = df.loc[after, cols] * 1.3

    elif scenario == "noise":
        # Compteur dégradé : bruit gaussien sur la cible
        std = df["global_active_power"].std()
        noise = rng.normal(0, 0.5 * std, after.sum())
        df.loc[after, "global_active_power"] = (
            df.loc[after, "global_active_power"] + noise
        ).clip(lower=0)

    elif scenario == "sensor_fault":
        # Capteur de tension bloqué sur sa moyenne
        df.loc[after, "voltage"] = df["voltage"].mean()

    else:
        raise ValueError(f"Scénario inconnu : {scenario} (choix : {SCENARIOS})")
    return df


def main(scenario):
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    d = cfg["datasets"]["household"]
    info = json.loads((ROOT / "data/processed/household_split_info.json").read_text(encoding="utf-8"))
    start = info["test"]["start"]

    raw = load_household(ROOT / cfg["paths"]["raw"] / d["raw_file"])
    hourly = clean_household(raw, cfg["resample_freq"], cfg["max_interp_gap_hours"])
    drifted = apply_drift(hourly, scenario, start)

    feats = build_features(
        drifted, d["target"], cfg["lags"], cfg["rolling_windows"],
        exog_lags=cfg["exog_lags"], country=d["country"],
    )
    feats = feats[feats.index >= pd.Timestamp(start)]
    feats = feats[[d["target"]] + load_feature_list("household")]   # même ordre que le train

    out = ROOT / "data" / "processed" / "drift"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"household_{scenario}.parquet"
    feats.to_parquet(path)
    print(f"{scenario} : {len(feats)} lignes à partir de {start} -> {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mean_shift")