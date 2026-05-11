"""
Módulo de configuración centralizado usando Pydantic.
Contiene los hiperparámetros y rutas para todo el proyecto.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class AppConfig(BaseModel):
    """
    Configuración principal del sistema para ML.
    """
    # Rutas de datos
    data_path: str = Field(
        default="icfes_final_serializado.parquet", 
        description="Ruta al dataset Parquet"
    )
    
    # Columnas importantes
    target_col: str = Field(default="punt_global", description="Variable objetivo")
    cat_col_to_encode: str = Field(
        default="cole_mcpio_ubicacion", 
        description="Columna categórica para aplicar Target Encoding"
    )
    
    # Hiperparámetros de la red neuronal MLP (PyTorch)
    mlp_hidden_dims: List[int] = Field(default_factory=lambda: [128, 64, 32])
    mlp_lr: float = Field(default=0.001)
    mlp_batch_size: int = Field(default=256)
    mlp_epochs: int = Field(default=100)
    mlp_patience: int = Field(default=10, description="Paciencia para el Early Stopping")
    
    # Hiperparámetros para XGBoost
    xgb_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "n_jobs": -1
        }
    )
    
    # Semilla aleatoria para reproducibilidad
    random_state: int = Field(default=42)

# Instancia global de configuración
config = AppConfig()
