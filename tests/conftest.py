"""Fixtures compartidos para los tests de saber_xai."""
import pytest

from saber_xai.config import config


@pytest.fixture(autouse=True)
def _allow_dummy_data():
    """Los datos dummy solo se permiten en tests (la carga real es estricta)."""
    original = config.allow_dummy_data
    config.allow_dummy_data = True
    yield
    config.allow_dummy_data = original
