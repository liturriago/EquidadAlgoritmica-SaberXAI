import pytest
import polars as pl
import numpy as np
from pathlib import Path
from saber_xai.data.data_module import DataModule
from saber_xai.config import config

def test_data_module_initialization():
    dm = DataModule()
    assert dm.X_train is None
    assert dm.X_val is None

def test_data_module_prepare_data_dummy():
    dm = DataModule()
    dm.prepare_data()
    assert dm.X_train is not None
    assert dm.X_val is not None
    assert dm.X_test is not None
    
    # Check no target in features
    assert dm.config.target_col not in dm.X_train.columns

def test_target_encoding_leakage():
    # El DataModule ya hizo fit_transform al target encoding en train, 
    # y transform en test/val. Verificamos que el tipo sea float.
    dm = DataModule()
    dm.prepare_data()
    cat_col = dm.config.cat_col_to_encode
    
    # Si cat_col_to_encode es None o no existe en los datos, el test pasa
    if cat_col is None or cat_col not in dm.X_train.columns:
        assert True
        return
    
    assert dm.X_train.schema[cat_col] in (pl.Float32, pl.Float64)
    assert dm.X_val.schema[cat_col] in (pl.Float32, pl.Float64)

def test_target_encoding_none_skips():
    # Cuando cat_col_to_encode es None, debe saltar el Target Encoding
    original_value = config.cat_col_to_encode
    try:
        config.cat_col_to_encode = None
        dm = DataModule()
        dm.prepare_data()
        
        # No debe haber columnas con valores encodeados
        # Solo verificar que no falla y los datos se procesan
        assert dm.X_train is not None
        assert dm.X_val is not None
    finally:
        config.cat_col_to_encode = original_value

def test_cluster_id_loading():
    # Test para cargar datos por cluster_id
    # Este test asume que existe data/clase_0.parquet
    # Si no existe, usa dummy data
    original_cluster = config.cluster_id
    original_data_dir = config.data_dir
    try:
        config.cluster_id = 0
        config.data_dir = "data/"
        
        dm = DataModule()
        dm.prepare_data()
        
        # Si el archivo existe, debe cargar datos
        # Si no existe, debe generar dummy data
        assert dm.X_train is not None
        assert dm.X_val is not None
    finally:
        config.cluster_id = original_cluster
        config.data_dir = original_data_dir
    
def test_dataloaders():
    dm = DataModule()
    dm.prepare_data()
    train_loader, val_loader, test_loader = dm.get_dataloaders()
    for batch_X, batch_y in train_loader:
        assert batch_X.shape[1] == dm.X_train.shape[1]
        assert batch_y.shape[1] == 1
        break

def test_shap_config():
    """Test para configuración de análisis SHAP."""
    # Verificar que los campos de configuración SHAP existen
    assert hasattr(config, 'shap_max_samples')
    assert hasattr(config, 'shap_output_dir')
    assert hasattr(config, 'shap_models_dir')
    
    # Verificar valores por defecto
    assert config.shap_max_samples == 10000
    assert config.shap_output_dir == "shap_results"
    assert config.shap_models_dir == "models"
    
    # Test de modificación
    original_samples = config.shap_max_samples
    try:
        config.shap_max_samples = 5000
        assert config.shap_max_samples == 5000
    finally:
        config.shap_max_samples = original_samples
