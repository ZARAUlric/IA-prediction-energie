import numpy as np
import pandas as pd
import holidays


def shift_exogenous(df, target, exog_lags=(1,)):
    """Remplace chaque variable non-cible par ses valeurs décalées.
    À l'instant t, on n'utilise que des mesures antérieures (pas de fuite)."""
    out = df[[target]].copy()
    for c in [c for c in df.columns if c != target]:
        for l in exog_lags:
            out[f"{c}_lag_{l}"] = df[c].shift(l)
    return out


def add_calendar(df, country="FR"):
    idx = df.index
    df["hour"] = idx.hour
    df["dayofweek"] = idx.dayofweek
    df["month"] = idx.month
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    hol = holidays.country_holidays(country, years=range(idx.year.min(), idx.year.max() + 1))
    hol_dates = pd.to_datetime(list(hol.keys()))
    df["is_holiday"] = idx.normalize().isin(hol_dates).astype(int)

    # Encodage cyclique
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def add_lags(df, target, lags):
    for l in lags:
        df[f"{target}_lag_{l}"] = df[target].shift(l)
    return df


def add_rolling(df, target, windows):
    s = df[target].shift(1)  # la fenêtre exclut la valeur courante
    for w in windows:
        df[f"{target}_rollmean_{w}"] = s.rolling(w).mean()
        df[f"{target}_rollstd_{w}"] = s.rolling(w).std()
    return df


def build_features(df, target, lags, windows, exog_lags=(1,), country="FR"):
    """L'index doit être régulier (sortie de resample) pour que shift() = décalage temporel."""
    df = shift_exogenous(df, target, exog_lags)
    df = add_calendar(df, country)
    df = add_lags(df, target, lags)
    df = add_rolling(df, target, windows)
    return df.dropna()