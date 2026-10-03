from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import numpy as np


class RandomForestModel(BaseEstimator):
    """Modèle Random Forest pour la prédiction de consommation énergétique"""
    
    def __init__(self, params=None):
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42,
                'n_jobs': -1
            }
        self.params = params
        self.model = None
    
    def fit(self, X, y):
        self.model = RandomForestRegressor(**self.params)
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """Retourne l'importance des features"""
        if self.model is None:
            raise ValueError("Modèle non entraîné")
        return dict(zip(self.model.feature_names_in_, self.model.feature_importances_))


def tune_random_forest(X_train, y_train, param_grid=None, cv_splits=5):
    """Recherche d'hyperparamètres avec validation croisée temporelle"""
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    
    base_model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1
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
