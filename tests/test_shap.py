"""Tests para el análisis SHAP."""
import pytest
import numpy as np
import polars as pl
from pathlib import Path
import tempfile
import shutil

from saber_xai.config import config


def test_shap_config_defaults():
    """Verifica que la configuración SHAP tiene valores por defecto correctos."""
    assert config.shap_max_samples == 10000
    assert config.shap_output_dir == "shap_results"
    assert config.shap_models_dir == "models"
    assert config.shap_data_dir == "data"
    assert config.shap_clusters == [0, 1, 2, 3, 4, 5, 6]
    assert config.shap_model_type == "both"


def test_shap_config_modification():
    """Verifica que se puede modificar la configuración SHAP."""
    original_samples = config.shap_max_samples
    original_model_type = config.shap_model_type
    original_data_dir = config.shap_data_dir
    original_clusters = config.shap_clusters.copy()
    try:
        config.shap_max_samples = 5000
        assert config.shap_max_samples == 5000
        
        config.shap_model_type = "xgb"
        assert config.shap_model_type == "xgb"
        
        config.shap_model_type = "mlp"
        assert config.shap_model_type == "mlp"
        
        config.shap_data_dir = "custom_data"
        assert config.shap_data_dir == "custom_data"
        
        config.shap_clusters = [0, 1, 2]
        assert config.shap_clusters == [0, 1, 2]
    finally:
        config.shap_max_samples = original_samples
        config.shap_model_type = original_model_type
        config.shap_data_dir = original_data_dir
        config.shap_clusters = original_clusters


def test_sample_data_function():
    """Test para la función de muestreo de datos."""
    # Importar la función desde el script
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from shap_analysis import sample_data
    
    # Datos de prueba
    X = np.random.randn(1000, 10)
    y = np.random.randn(1000)
    
    # Muestrear a 100
    X_sampled, y_sampled = sample_data(X, y, max_samples=100, random_state=42)
    
    assert X_sampled.shape[0] == 100
    assert y_sampled.shape[0] == 100
    assert X_sampled.shape[1] == 10
    
    # Si max_samples >= len(X), no debe muestrear
    X_full, y_full = sample_data(X, y, max_samples=2000, random_state=42)
    assert X_full.shape[0] == 1000
    assert y_full.shape[0] == 1000


def test_load_cluster_data_function():
    """Test para la función de carga de datos por clúster."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from shap_analysis import load_cluster_data
    
    # Crear datos de prueba temporales
    temp_dir = tempfile.mkdtemp()
    try:
        # Crear un parquet de prueba
        df = pl.DataFrame({
            "punt_global": np.random.randn(100).astype(np.float32),
            "feature1": np.random.randn(100).astype(np.float32),
            "feature2": np.random.randn(100).astype(np.float32),
            "category": ["A", "B"] * 50
        })
        df.write_parquet(Path(temp_dir) / "clase_0.parquet")
        
        # Cargar datos
        X, y, feature_names = load_cluster_data(Path(temp_dir), cluster_id=0)
        
        assert X.shape[0] == 100
        assert y.shape[0] == 100
        assert len(feature_names) >= 2  # Al menos feature1 y feature2
        assert "punt_global" not in feature_names
        
    finally:
        shutil.rmtree(temp_dir)


def test_load_cluster_data_not_found():
    """Test para error cuando no existe el archivo."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from shap_analysis import load_cluster_data
    
    with pytest.raises(FileNotFoundError):
        load_cluster_data(Path("/nonexistent"), cluster_id=99)


def test_compute_shap_xgb_function():
    """Test para la función de cálculo SHAP con XGBoost."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from shap_analysis import compute_shap_xgb
    
    # Crear modelo XGBoost de prueba
    import xgboost as xgb
    
    # Datos de entrenamiento simples
    X_train = np.random.randn(100, 5)
    y_train = np.random.randn(100)
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {"max_depth": 3, "eta": 0.1, "objective": "reg:squarederror"}
    model = xgb.train(params, dtrain, num_boost_round=10)
    
    # Guardar modelo temporalmente
    temp_dir = tempfile.mkdtemp()
    try:
        model_path = Path(temp_dir) / "test_model.json"
        model.save_model(str(model_path))
        
        # Calcular SHAP
        X_test = np.random.randn(20, 5)
        shap_values, feature_names = compute_shap_xgb(model_path, X_test, [f"feat_{i}" for i in range(5)])
        
        assert shap_values.shape == (20, 5)
        assert len(feature_names) == 5
        assert not np.isnan(shap_values).any()
        
    finally:
        shutil.rmtree(temp_dir)


def test_compute_shap_mlp_function():
    """Test para la función de cálculo SHAP con MLP."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from shap_analysis import compute_shap_mlp
    import torch
    from saber_xai.models.mlp_model import MLP
    
    # Crear modelo MLP de prueba
    input_dim = 5
    model = MLP(input_dim=input_dim)
    
    # Guardar modelo temporalmente
    temp_dir = tempfile.mkdtemp()
    try:
        model_path = Path(temp_dir) / "test_mlp.pth"
        torch.save(model.state_dict(), model_path)
        
        # Calcular SHAP
        X_test = np.random.randn(20, 5).astype(np.float32)
        shap_values, feature_names = compute_shap_mlp(model_path, X_test, input_dim, [f"feat_{i}" for i in range(5)])
        
        assert shap_values.shape == (20, 5)
        assert len(feature_names) == 5
        assert not np.isnan(shap_values).any()
        
    finally:
        shutil.rmtree(temp_dir)
