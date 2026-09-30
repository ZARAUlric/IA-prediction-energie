import numpy as np
import pytest

from src.data.loader import (
    load_dataset, load_feature_list, load_scaler, get_target,
)

DATASETS = ["household", "appliances"]

# Mesures de la même heure qui ne doivent PAS être dans les features
FORBIDDEN = {
    "household": ["global_intensity", "voltage", "sub_metering_1", "sub_metering_rest",
                  "global_reactive_power"],
    "appliances": ["t1", "rh_1", "t_out", "lights"],
}


def splits(name):
    return [load_dataset(name, s) for s in ("train", "val", "test")]


@pytest.mark.parametrize("name", DATASETS)
def test_no_nan(name):
    for df in splits(name):
        assert df.isna().sum().sum() == 0


@pytest.mark.parametrize("name", DATASETS)
def test_sorted_index(name):
    for df in splits(name):
        assert df.index.is_monotonic_increasing


@pytest.mark.parametrize("name", DATASETS)
def test_no_overlap(name):
    tr, va, te = splits(name)
    assert tr.index.max() < va.index.min()
    assert va.index.max() < te.index.min()


@pytest.mark.parametrize("name", DATASETS)
def test_same_columns(name):
    tr, va, te = splits(name)
    assert list(tr.columns) == list(va.columns) == list(te.columns)


@pytest.mark.parametrize("name", DATASETS)
def test_feature_list_matches_columns(name):
    tr = load_dataset(name, "train")
    target = get_target(name)
    assert load_feature_list(name) == [c for c in tr.columns if c != target]


@pytest.mark.parametrize("name", DATASETS)
def test_scaler_fitted_on_train_only(name):
    tr = load_dataset(name, "train")
    feats = load_feature_list(name)
    assert np.allclose(load_scaler(name).mean_, tr[feats].mean().to_numpy())


@pytest.mark.parametrize("name", DATASETS)
def test_no_leakage_lag(name):
    import pandas as pd
    full = pd.concat(splits(name))
    t = get_target(name)
    prev = full[t].shift(1, freq="1h").reindex(full.index)   # valeur de l'heure précédente
    mask = prev.notna()
    assert mask.sum() > 0
    assert np.allclose(full.loc[mask, f"{t}_lag_1"], prev[mask])


@pytest.mark.parametrize("name", DATASETS)
def test_exogenous_are_lagged(name):
    cols = load_dataset(name, "train").columns
    for c in FORBIDDEN[name]:
        assert c not in cols, f"{c} (même heure) est présent : fuite de données"