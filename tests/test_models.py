import pytest
import torch
import numpy as np
import xgboost as xgb
from saber_xai.models.mlp_model import MLP
from saber_xai.models.xgb_model import XGBoostModel
from saber_xai.models.evaluator import ModelEvaluator

def test_mlp_forward():
    input_dim = 10
    model = MLP(input_dim=input_dim)
    dummy_input = torch.randn(32, input_dim) # batch_size 32
    output = model(dummy_input)
    assert output.shape == (32, 1)

def test_xgboost_init():
    model = XGBoostModel()
    assert model.model is None

def test_evaluator_compute_metrics():
    evaluator = ModelEvaluator()
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([110, 190, 310, 390, 510])
    
    metrics = evaluator._compute_metrics(y_true, y_pred)
    
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert "mbe" in metrics
    
    # MBE debe ser positivo (sobreestima ligeramente)
    assert metrics["mbe"] > 0
    # RMSE debe ser mayor que MAE
    assert metrics["rmse"] >= metrics["mae"]

def test_evaluator_evaluate_by_cluster():
    evaluator = ModelEvaluator()
    
    # Datos sintéticos con 3 clústeres
    n_samples = 300
    y_true = np.random.uniform(200, 400, n_samples)
    y_pred = y_true + np.random.normal(0, 10, n_samples)
    cluster_labels = np.repeat([0, 1, 2], 100)
    
    results = evaluator.evaluate_by_cluster(y_true, y_pred, cluster_labels, "TestModel")
    
    # Debe retornar métricas para cada clúster
    assert len(results) == 3
    assert 0 in results
    assert 1 in results
    assert 2 in results
    
    # Cada clúster debe tener las 4 métricas
    for cluster_id in [0, 1, 2]:
        assert "rmse" in results[cluster_id]
        assert "mae" in results[cluster_id]
        assert "r2" in results[cluster_id]
        assert "mbe" in results[cluster_id]

def test_evaluator_evaluate_by_cluster_empty():
    evaluator = ModelEvaluator()
    
    # Clúster con solo un valor
    y_true = np.array([100, 200, 300])
    y_pred = np.array([105, 195, 305])
    cluster_labels = np.array([0, 0, 0])
    
    results = evaluator.evaluate_by_cluster(y_true, y_pred, cluster_labels, "TestModel")
    
    assert len(results) == 1
    assert 0 in results
