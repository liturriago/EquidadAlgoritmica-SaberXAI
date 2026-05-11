"""
Modelo XGBoost con Cross-Validation.
"""
import xgboost as xgb
import numpy as np
from sklearn.model_selection import KFold
from saber_xai.config import config
import json
import os

class XGBoostModel:
    """Clase envoltorio para entrenamiento y gestión del modelo XGBoost."""
    def __init__(self):
        self.config = config
        self.model = None

    def train_cv(self, dtrain: xgb.DMatrix, n_splits: int = 5):
        """Entrena el modelo usando validación cruzada y retorna los resultados."""
        print(f"Entrenando XGBoost con {n_splits}-fold CV...")
        cv_results = xgb.cv(
            dtrain=dtrain,
            params=self.config.xgb_params,
            nfold=n_splits,
            num_boost_round=self.config.xgb_params.get("n_estimators", 100),
            early_stopping_rounds=10,
            metrics="rmse",
            as_pandas=True,
            seed=self.config.random_state
        )
        print("CV Finalizado. Mejor RMSE:")
        print(cv_results.tail(1))
        
        # Entrenar en todo el train set para guardar el modelo final
        best_num_boost_round = len(cv_results)
        self.model = xgb.train(
            params=self.config.xgb_params,
            dtrain=dtrain,
            num_boost_round=best_num_boost_round
        )
        return cv_results

    def save_model(self, path: str = "xgboost_model.json"):
        """Guarda el modelo entrenado en formato json o joblib."""
        if self.model:
            self.model.save_model(path)
            print(f"Modelo guardado en {path}")
        else:
            print("El modelo no ha sido entrenado.")
