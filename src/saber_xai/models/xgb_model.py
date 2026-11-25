"""
Modelo XGBoost con Early Stopping sobre conjunto de validación.
"""
import json
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

    def predict(self, dmatrix: xgb.DMatrix) -> "xgb.ArrayLike":
        """Predice usando la mejor ronda del early stopping.

        Sin esto, el Booster nativo usaría todas las rondas entrenadas en lugar
        de las de mejor validación (best_iteration), lo que altera las métricas
        de test y el MBE cuando el early stopping no se activó.
        """
        if self.model is None:
            raise RuntimeError("El modelo no ha sido entrenado.")
        best_iter = getattr(self.model, "best_iteration", None)
        if best_iter is not None:
            return self.model.predict(dmatrix, iteration_range=(0, best_iter + 1))
        return self.model.predict(dmatrix)

    def save_model(self, path: str = "xgboost_model.json", feature_names: list = None):
        """Guarda el modelo entrenado en formato json más un sidecar de metadatos.

        El sidecar ``{path}.meta.json`` registra los nombres de features y la
        mejor ronda: resuelve la discrepancia de conteos de features entre el
        artículo y los modelos reales (verificación independiente por corrida).
        """
        if self.model:
            self.model.save_model(path)
            print(f"Modelo guardado en {path}")
            meta = {
                "best_iteration": getattr(self.model, "best_iteration", None),
                "best_score": getattr(self.model, "best_score", None),
                "params": self.config.xgb_params,
                "n_features": len(feature_names) if feature_names is not None else None,
                "feature_names": feature_names,
            }
            meta_path = f"{path}.meta.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            print(f"Metadatos del modelo guardados en {meta_path}")
        else:
            print("El modelo no ha sido entrenado.")
