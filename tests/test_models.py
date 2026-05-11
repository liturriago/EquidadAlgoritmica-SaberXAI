import pytest
import torch
import xgboost as xgb
from src.models.mlp_model import MLP
from src.models.xgb_model import XGBoostModel

def test_mlp_forward():
    input_dim = 10
    model = MLP(input_dim=input_dim)
    dummy_input = torch.randn(32, input_dim) # batch_size 32
    output = model(dummy_input)
    assert output.shape == (32, 1)

def test_xgboost_init():
    model = XGBoostModel()
    assert model.model is None
