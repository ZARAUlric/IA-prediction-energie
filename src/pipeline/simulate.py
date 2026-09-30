import numpy as np
import pandas as pd


def simulate_available_data(df, cutoff):
    """Ne garde que les données disponibles jusqu'à `cutoff` (None = tout)."""
    if cutoff is None:
        return df
    return df[df.index <= pd.Timestamp(cutoff)]


def make_cutoffs(df, n_runs=4, start_frac=0.5):
    """n_runs dates régulièrement espacées entre start_frac et la fin de la période."""
    idx = df.index
    fracs = np.linspace(start_frac, 1.0, n_runs)
    return [idx[int(f * (len(idx) - 1))] for f in fracs]