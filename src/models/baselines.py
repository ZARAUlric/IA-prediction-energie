import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class MeanBaseline(BaseEstimator):
    """Baseline : prédit la moyenne de la cible sur le train"""
    
    def __init__(self):
        self.mean_value = None
    
    def fit(self, X, y):
        self.mean_value = y.mean()
        return self
    
    def predict(self, X):
        return np.full(len(X), self.mean_value)


class PersistenceBaseline(BaseEstimator):
    """Baseline : prédit la dernière valeur connue (lag 1)"""
    
    def __init__(self):
        self.lag_1_col = None
    
    def fit(self, X, y):
        # Trouver la colonne lag_1 de la cible
        for col in X.columns:
            if "lag_1" in col and "global_active_power" in col or "appliances" in col:
                self.lag_1_col = col
                break
        if self.lag_1_col is None:
            # Fallback : prendre la première colonne lag_1
            lag_cols = [c for c in X.columns if "lag_1" in c]
            if lag_cols:
                self.lag_1_col = lag_cols[0]
            else:
                raise ValueError("Aucune colonne lag_1 trouvée")
        return self
    
    def predict(self, X):
        return X[self.lag_1_col].values


class RollingMeanBaseline(BaseEstimator):
    """Baseline : prédit la moyenne glissante sur 24h"""
    
    def __init__(self, window=24):
        self.window = window
        self.rollmean_col = None
    
    def fit(self, X, y):
        # Trouver la colonne rollmean_24
        for col in X.columns:
            if f"rollmean_{self.window}" in col:
                self.rollmean_col = col
                break
        if self.rollmean_col is None:
            raise ValueError(f"Aucune colonne rollmean_{self.window} trouvée")
        return self
    
    def predict(self, X):
        return X[self.rollmean_col].values
