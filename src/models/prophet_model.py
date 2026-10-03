import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.base import BaseEstimator


class ProphetModel(BaseEstimator):
    """Modèle Prophet pour la prédiction de consommation énergétique"""
    
    def __init__(self, params=None):
        if params is None:
            params = {
                'yearly_seasonality': True,
                'weekly_seasonality': True,
                'daily_seasonality': True,
                'seasonality_mode': 'multiplicative',
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'holidays_prior_scale': 10.0
            }
        self.params = params
        self.model = None
        self.extra_regressors = None
    
    def fit(self, X, y):
        # Préparer les données pour Prophet (format ds, y)
        df = pd.DataFrame({
            'ds': X.index,
            'y': y.values
        })
        
        # Ajouter les régresseurs externes (variables calendaires)
        self.extra_regressors = []
        for col in ['hour', 'dayofweek', 'month', 'is_weekend', 'is_holiday']:
            if col in X.columns:
                df[col] = X[col].values
                self.extra_regressors.append(col)
        
        # Créer et ajuster le modèle
        self.model = Prophet(**self.params)
        
        # Ajouter les régresseurs
        for reg in self.extra_regressors:
            self.model.add_regressor(reg, mode='additive')
        
        self.model.fit(df)
        return self
    
    def predict(self, X):
        # Préparer les données de prédiction
        df = pd.DataFrame({'ds': X.index})
        
        # Ajouter les régresseurs
        for reg in self.extra_regressors:
            if reg in X.columns:
                df[reg] = X[reg].values
        
        # Prédire
        forecast = self.model.predict(df)
        return forecast['yhat'].values
    
    def get_feature_importance(self):
        """Prophet n'a pas d'importance de features classique"""
        return {"note": "Prophet utilise une décomposition additive/multiplicative, pas d'importance de features"}
