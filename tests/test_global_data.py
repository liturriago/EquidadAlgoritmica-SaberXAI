"""Tests del modo global del DataModule (línea de base global).

Validan la propiedad científica clave: el modelo global se entrena y evalúa
sobre la UNIÓN de los splits por clúster, es decir, sobre exactamente las
mismas filas de test que los modelos por clúster (comparación válida de
MBE/R²/RMSE para la Tabla 4).
"""
import numpy as np
import polars as pl
import pytest
from pathlib import Path

from saber_xai.config import config
from saber_xai.data.data_module import DataModule


CONFIG_FIELDS = (
    "data_dir", "data_path", "cluster_id", "cat_col_to_encode",
    "territory_indicators", "allow_dummy_data",
)


@pytest.fixture
def cluster_dir(tmp_path):
    """Crea clase_0/clase_1.parquet con target único por fila (id de fila)."""
    for cid, n in ((0, 200), (1, 300)):
        rng = np.random.default_rng(cid)
        df = pl.DataFrame({
            # Valores únicos por fila en todo el dataset: sirven como id
            "punt_global": np.arange(n, dtype=np.float64) + cid * 1000,
            "fami_estrato": rng.integers(1, 7, n),
            "estu_edad": rng.uniform(15, 60, n),
            "cole_area_ubicacion": np.where(np.arange(n) % 2 == 0, "Rural", "Urbano"),
        })
        df.write_parquet(tmp_path / f"clase_{cid}.parquet")
    return tmp_path


@pytest.fixture
def config_ctx(cluster_dir):
    """Apunta la config a los parquet sintéticos y restaura al finalizar."""
    saved = {k: getattr(config, k) for k in CONFIG_FIELDS}
    config.data_dir = str(cluster_dir)
    config.data_path = "no_existe.parquet"
    config.cluster_id = None
    config.cat_col_to_encode = None
    config.territory_indicators = False
    config.allow_dummy_data = False
    yield config
    for k, v in saved.items():
        setattr(config, k, v)


def _test_y_values(dm: DataModule) -> set:
    return set(np.round(dm.y_test.to_numpy().flatten(), 6).tolist())


def test_global_split_matches_per_cluster_splits(config_ctx):
    """El test del modo global debe ser la unión exacta de los tests por clúster."""
    per_cluster_test = set()
    per_cluster_counts = {}
    for cid in (0, 1):
        config_ctx.cluster_id = cid
        dm_c = DataModule()
        dm_c.prepare_data()
        per_cluster_test |= _test_y_values(dm_c)
        per_cluster_counts[cid] = len(dm_c.y_test)

    config_ctx.cluster_id = None
    dm_g = DataModule()
    dm_g.prepare_data()

    assert _test_y_values(dm_g) == per_cluster_test
    assert dm_g.split_meta["n_test"] == sum(per_cluster_counts.values())


def test_global_row_counts_add_up(config_ctx):
    """Los tamaños de split globales son la suma de los splits por clúster."""
    counts = []
    for cid in (0, 1):
        config_ctx.cluster_id = cid
        dm_c = DataModule()
        dm_c.prepare_data()
        counts.append((len(dm_c.X_train), len(dm_c.X_val), len(dm_c.X_test)))

    config_ctx.cluster_id = None
    dm_g = DataModule()
    dm_g.prepare_data()

    assert dm_g.split_meta["n_train"] == sum(c[0] for c in counts)
    assert dm_g.split_meta["n_val"] == sum(c[1] for c in counts)
    assert dm_g.split_meta["n_test"] == sum(c[2] for c in counts)


def test_test_cluster_labels_alignment(config_ctx):
    """test_cluster_labels alinea cada fila de test con su clúster LCA."""
    config_ctx.cluster_id = None
    dm = DataModule()
    dm.prepare_data()

    labels = dm.test_cluster_labels
    assert labels is not None
    assert len(labels) == len(dm.y_test)
    # 2 clústeres sintéticos → ambas etiquetas presentes
    assert set(np.unique(labels).tolist()) == {0, 1}


def test_territory_indicators(config_ctx):
    """territory_indicators añade one-hot de clase_lca (drop_first) y la elimina."""
    config_ctx.territory_indicators = True
    dm = DataModule()
    dm.prepare_data()

    indicator_cols = [c for c in dm.X_train.columns if c.startswith("clase_lca_")]
    assert dm.TERRITORY_COL not in dm.X_train.columns
    # 2 clústeres con drop_first → 1 indicadora
    assert len(indicator_cols) == 1
    # Las indicadoras existen también en val/test y están alineadas
    for df in (dm.X_val, dm.X_test):
        for col in indicator_cols:
            assert col in df.columns


def test_without_indicators_drops_territory_column(config_ctx):
    """Sin indicadoras, la columna cruda del clúster no queda en las features."""
    dm = DataModule()
    dm.prepare_data()

    assert dm.TERRITORY_COL not in dm.X_train.columns
    assert not any(c.startswith("clase_lca_") for c in dm.X_train.columns)


def test_strict_mode_raises_on_missing_data(config_ctx):
    """Sin allow_dummy_data, la ausencia de parquet lanza FileNotFoundError."""
    config_ctx.data_dir = str(Path(config_ctx.data_dir) / "no_existe")
    config_ctx.allow_dummy_data = False

    dm = DataModule()
    with pytest.raises(FileNotFoundError):
        dm.prepare_data()


def test_dummy_fallback_only_when_allowed(config_ctx):
    """Con allow_dummy_data=True, la carga fallida genera datos sintéticos."""
    config_ctx.data_dir = str(Path(config_ctx.data_dir) / "no_existe")
    config_ctx.allow_dummy_data = True

    dm = DataModule()
    dm.prepare_data()
    assert dm.X_train is not None
    assert len(dm.X_train) > 0


def test_split_meta_and_save(config_ctx, tmp_path):
    """split_meta documenta modo, tamaños, medias y features; y se persiste."""
    config_ctx.cluster_id = None
    dm = DataModule()
    dm.prepare_data()

    meta = dm.split_meta
    assert meta["mode"] == "global"
    assert meta["seed"] == config.random_state
    assert meta["n_features"] == len(meta["feature_names"])
    assert dm.TERRITORY_COL not in meta["feature_names"]
    for part in ("train", "val", "test"):
        assert meta[f"n_{part}"] > 0
        assert np.isfinite(meta[f"y_mean_{part}"])

    out = tmp_path / "split_meta.json"
    dm.save_split_meta(str(out))
    assert out.exists()
