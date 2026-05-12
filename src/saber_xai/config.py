"""
Módulo de configuración centralizado usando Pydantic.
Contiene los hiperparámetros y rutas para todo el proyecto.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import yaml

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
    xgb_n_estimators: int = Field(default=200, description="Número de árboles (num_boost_round)")
    xgb_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "max_depth": 6,
            "learning_rate": 0.05,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "nthread": -1
        }
    )
    
    # Semilla aleatoria para reproducibilidad
    random_state: int = Field(default=42)

    def load_from_yaml(self, yaml_path: str) -> None:
        """Carga parámetros desde un archivo YAML y actualiza la configuración actual."""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        if data:
            for key, value in data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    print(f"Advertencia: El parámetro '{key}' en el archivo YAML no está definido en AppConfig.")

# Instancia global de configuración
config = AppConfig()
