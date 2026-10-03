from pathlib import Path

import yaml
from prefect import flow, task, get_run_logger

from src.data.clean import (
    load_household, clean_household, load_appliances, clean_appliances,
)
from src.features.build import build_features
from src.features.split import temporal_split, fit_scaler, save_splits, save_artifacts
from src.pipeline.simulate import simulate_available_data

ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path):
    p = Path(config_path)
    if not p.is_absolute():
        p = ROOT / p
    return yaml.safe_load(p.read_text(encoding="utf-8"))


@task
def ingest(name, cfg):
    path = ROOT / cfg["paths"]["raw"] / cfg["datasets"][name]["raw_file"]
    df = load_household(path) if name == "household" else load_appliances(path)
    get_run_logger().info(f"[{name}] ingest : {len(df)} lignes ({df.index.min()} -> {df.index.max()})")
    return df


@task
def clean(name, df, cfg):
    if name == "household":
        out = clean_household(df, cfg["resample_freq"], cfg["max_interp_gap_hours"])
    else:
        out = clean_appliances(df, cfg["resample_freq"])
    get_run_logger().info(f"[{name}] clean : {len(out)} lignes, {out.isna().mean().mean():.2%} de NaN")
    return out


@task
def restrict(name, df, cutoff):
    out = simulate_available_data(df, cutoff)
    get_run_logger().info(f"[{name}] cutoff={cutoff} : {len(out)} lignes conservées")
    return out


@task
def features(name, df, cfg):
    d = cfg["datasets"][name]
    out = build_features(
        df, d["target"], cfg["lags"], cfg["rolling_windows"],
        exog_lags=cfg["exog_lags"], country=d["country"],
    )
    get_run_logger().info(f"[{name}] features : {out.shape[1]} colonnes, {len(out)} lignes")
    return out


@task
def split_and_save(name, df, cfg, out_dir):
    target = cfg["datasets"][name]["target"]
    train, val, test = temporal_split(df, **cfg["split"])
    feats = [c for c in df.columns if c != target]
    scaler = fit_scaler(train, feats)

    proc_dir = Path(out_dir) if out_dir else ROOT / cfg["paths"]["processed"]
    art_dir = Path(out_dir) if out_dir else ROOT / cfg["paths"]["models"]
    save_splits(train, val, test, name, proc_dir)
    save_artifacts(name, scaler, feats, art_dir)

    get_run_logger().info(f"[{name}] split : train={len(train)} val={len(val)} test={len(test)}")
    return len(train), len(val), len(test)


@flow(name="data-pipeline")
def data_pipeline(datasets=None, config_path="configs/config.yaml",
                  cutoff_date=None, out_dir=None, train_fn=None):
    cfg = load_config(config_path)
    names = datasets or list(cfg["datasets"])
    cutoff = cutoff_date or cfg.get("cutoff_date")

    results = {}
    for name in names:
        raw = ingest(name, cfg)
        cleaned = clean(name, raw, cfg)
        available = restrict(name, cleaned, cutoff)
        feats = features(name, available, cfg)
        results[name] = split_and_save(name, feats, cfg, out_dir)
        if train_fn is not None and out_dir is None:
            train_fn(name)          # fonction fournie par P2/P3, loggue dans MLflow
    return results


if __name__ == "__main__":
    data_pipeline()