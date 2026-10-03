import xgboost as xgb
from sklearn.base import BaseEstimator
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import numpy as np


class XGBoostModel(BaseEstimator):
    """Modèle XGBoost pour la prédiction de consommation énergétique"""
    
    def __init__(self, params=None):
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'objective': 'reg:squarederror'
            }
        self.params = params
        self.model = None
    
    def fit(self, X, y):
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """Retourne l'importance des features"""
        if self.model is None:
            raise ValueError("Modèle non entraîné")
        return dict(zip(self.model.feature_names_in_, self.model.feature_importances_))


def tune_xgboost(X_train, y_train, param_grid=None, cv_splits=5):
    """Recherche d'hyperparamètres avec validation croisée temporelle"""
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0]
        }
    
    base_model = xgb.XGBRegressor(
        random_state=42,
        objective='reg:squarederror'
    )
    
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': -grid_search.best_score_,
        'best_model': grid_search.best_estimator_
    }
