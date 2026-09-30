import numpy as np
import pandas as pd


# ---------- Chargement ----------

def load_household(path):
    df = pd.read_csv(path, sep=";", na_values=["?"], low_memory=False)
    df["timestamp"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"
    )
    df = df.drop(columns=["Date", "Time"]).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.columns = [c.lower() for c in df.columns]
    return df.astype(float)


def load_appliances(path):
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return df


# ---------- Utilitaires ----------

def interpolate_short_gaps(df, max_gap):
    """Interpole uniquement les trous de longueur <= max_gap (en pas de temps).
    Les trous plus longs restent entièrement NaN."""
    filled = df.interpolate(method="time", limit_area="inside")
    for col in df.columns:
        na = df[col].isna()
        groups = (na != na.shift()).cumsum()
        gap_size = na.astype(int).groupby(groups).transform("sum")
        filled.loc[na & (gap_size > max_gap), col] = np.nan
    return filled


def flag_outliers(s, k=3.0):
    """Renvoie un masque booléen des valeurs hors [Q1 - k*IQR, Q3 + k*IQR].
    Sert à l'analyse : on ne supprime PAS ces pics (consommations réelles)."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


# ---------- Nettoyage ----------

def clean_household(df, freq="1h", max_gap_hours=3, min_valid_ratio=0.5):
    df = df.copy()

    # Valeurs physiquement impossibles -> NaN
    df.loc[df["global_active_power"] < 0, "global_active_power"] = np.nan
    df.loc[df["voltage"] <= 0, "voltage"] = np.nan

    # Énergie (Wh/min) non mesurée par les 3 compteurs
    sub_cols = ["sub_metering_1", "sub_metering_2", "sub_metering_3"]
    df["sub_metering_rest"] = (
        df["global_active_power"] * 1000 / 60
        - df[sub_cols].sum(axis=1, skipna=False)
    )

    # Rééchantillonnage : moyenne partout (robuste aux minutes manquantes)
    hourly = df.resample(freq).mean()

    # Heure invalide si trop peu de minutes valides
    minutes_per_period = pd.Timedelta(freq) / pd.Timedelta(minutes=1)
    valid = df["global_active_power"].resample(freq).count()
    hourly.loc[valid < min_valid_ratio * minutes_per_period] = np.nan

    # Les sub_metering sont en Wh/min -> Wh par période
    for c in sub_cols + ["sub_metering_rest"]:
        hourly[c] = hourly[c] * minutes_per_period

    # global_active_power reste en kW moyen sur l'heure (= kWh sur l'heure)
    max_gap = int(max_gap_hours * pd.Timedelta("1h") / pd.Timedelta(freq))
    return interpolate_short_gaps(hourly, max_gap)


def clean_appliances(df, freq="1h", min_valid_ratio=0.5):
    df = df.drop(columns=["rv1", "rv2"], errors="ignore").copy()
    df.columns = [c.lower() for c in df.columns]
    df = df[~df.index.duplicated(keep="first")]

    sum_cols = ["appliances", "lights"]
    mean_cols = [c for c in df.columns if c not in sum_cols]
    agg = {**{c: "sum" for c in sum_cols}, **{c: "mean" for c in mean_cols}}
    hourly = df.resample(freq).agg(agg)

    # Périodes partielles (début/fin du dataset) -> NaN
    expected = pd.Timedelta(freq) / pd.Timedelta(minutes=10)
    counts = df["appliances"].resample(freq).count()
    hourly.loc[counts < min_valid_ratio * expected] = np.nan
    return hourly