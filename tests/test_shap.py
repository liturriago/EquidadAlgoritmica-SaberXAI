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


def test_shap_config_modification():
    """Verifica que se puede modificar la configuración SHAP."""
    original = config.shap_max_samples
    try:
        config.shap_max_samples = 5000
        assert config.shap_max_samples == 5000
    finally:
        config.shap_max_samples = original


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
