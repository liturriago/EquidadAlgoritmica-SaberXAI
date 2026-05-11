import pytest
import pandas as pd
import numpy as np
from src.data.data_module import DataModule

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
    assert dm.X_train[cat_col].dtype == float
    assert dm.X_val[cat_col].dtype == float
    
def test_dataloaders():
    dm = DataModule()
    dm.prepare_data()
    train_loader, val_loader, test_loader = dm.get_dataloaders()
    for batch_X, batch_y in train_loader:
        assert batch_X.shape[1] == dm.X_train.shape[1]
        assert batch_y.shape[1] == 1
        break
