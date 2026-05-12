"""
Modelo XGBoost con Early Stopping sobre conjunto de validación.
"""
import xgboost as xgb
from saber_xai.config import config

class XGBoostModel:
    """Clase envoltorio para entrenamiento y gestión del modelo XGBoost."""
    def __init__(self):
        self.config = config
        self.model = None

    def train_with_early_stopping(self, dtrain: xgb.DMatrix, dval: xgb.DMatrix) -> None:
        """Entrena XGBoost con early stopping sobre el conjunto de validación.
        
        Evita xgb.cv (que crea N copias internas del DMatrix) y usa en cambio
        una watchlist con el dval ya disponible, que es mucho más eficiente en RAM.
        """
        watchlist = [(dtrain, 'train'), (dval, 'val')]
        num_boost_round = self.config.xgb_n_estimators

        print(f"Entrenando XGBoost (hasta {num_boost_round} rondas, early stopping en 10)...")
        self.model = xgb.train(
            params=self.config.xgb_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=watchlist,
            early_stopping_rounds=10,
            verbose_eval=25,
        )
        print(f"Mejor ronda: {self.model.best_iteration} | Mejor val-rmse: {self.model.best_score:.4f}")

    def save_model(self, path: str = "xgboost_model.json"):
        """Guarda el modelo entrenado en formato json o joblib."""
        if self.model:
            self.model.save_model(path)
            print(f"Modelo guardado en {path}")
        else:
            print("El modelo no ha sido entrenado.")
