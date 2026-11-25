"""Regresión del bug de one-hot por frame (to_dummies + alineación).

Cuando el nivel de REFERENCIA de train (el primero tras drop_first) no aparece
en val/test, el to_dummies por frame eliminaba un nivel distinto y las filas del
segundo nivel quedaban codificadas como el nivel de referencia (todos ceros),
corrompiendo las features de val/test de forma distinta entre splits (inflaba el
MBE con signo variable por clúster: el histórico −10,41 del clúster 6 era este
artefacto). El fix usa el esquema de one-hot de train para codificar val/test.
"""
import numpy as np
import polars as pl
import pytest

from saber_xai.config import config
from saber_xai.data.data_module import DataModule

CONFIG_FIELDS = (
    "data_dir", "data_path", "cluster_id", "cat_col_to_encode",
    "territory_indicators", "allow_dummy_data", "random_state",
)


@pytest.fixture
def config_ctx(tmp_path):
    saved = {k: getattr(config, k) for k in CONFIG_FIELDS}
    config.data_path = "no_existe.parquet"
    config.cluster_id = None
    config.data_dir = None
    config.cat_col_to_encode = None
    config.territory_indicators = False
    config.allow_dummy_data = False
    yield config
    for k, v in saved.items():
        setattr(config, k, v)


def _train_row_ids(df: pl.DataFrame, seed: int) -> set:
    """Replica el shuffle/split del pipeline y retorna los row_id de train."""
    n = len(df)
    shuffled = df.with_row_index("row_id").sample(fraction=1.0, shuffle=True, seed=seed)
    t = int(n * 0.15)
    v = int(n * 0.15)
    return set(shuffled.tail(n - t - v)["row_id"].to_list())


def test_reference_level_missing_in_val_test(config_ctx, tmp_path):
    """El nivel de referencia de train (ausente en val/test) no debe corromper
    el encoding de las filas de val/test."""
    rng = np.random.default_rng(7)
    n = 2000
    path = tmp_path / "datos.parquet"

    # Base: todas las filas con nivel 'B' (se escribe y se relee para que la
    # determinación de membresía use exactamente el mismo frame que el pipeline)
    pl.DataFrame({
        "punt_global": rng.uniform(150, 400, n),
        "cat": pl.Series(["B"] * n, dtype=pl.Categorical),
        "estu_edad": rng.uniform(15, 25, n),
    }).write_parquet(path)

    # Determinar la membresía y el orden de aparición del slice de train con la
    # misma semilla del pipeline: drop_first elimina el primer nivel APARECIDO en
    # el frame, así que 'A' se asigna a la primera fila del slice de train para
    # garantizar que 'A' es el nivel de referencia que train elimina
    df = pl.read_parquet(path)
    seed = config.random_state
    shuffled = df.with_row_index("row_id").sample(fraction=1.0, shuffle=True, seed=seed)
    t = int(n * 0.15)
    v = int(n * 0.15)
    train_part = shuffled.tail(n - t - v)
    train_ids = set(train_part["row_id"].to_list())
    first_train_row = int(train_part["row_id"][0])
    a_ids = {first_train_row} | set(sorted(train_ids - {first_train_row})[:49])

    # El nivel 'A' (referencia) existirá SOLO en filas de train
    df = df.with_columns(
        pl.Series("cat", ["A" if i in a_ids else "B" for i in range(n)],
                  dtype=pl.Categorical)
    )
    df.write_parquet(path)

    # Precondición: la determinación de membresía debe seguir siendo válida
    assert a_ids <= _train_row_ids(pl.read_parquet(path), seed=seed), (
        "el shuffle de polars cambió: revisar la determinación de membresía"
    )

    config_ctx.data_path = str(path)
    dm = DataModule()
    dm.prepare_data()

    # train tiene ambos niveles → una columna dummy ('A' fue el nivel eliminado)
    cols = [c for c in dm.X_train.columns if c.startswith("cat_")]
    assert cols == ["cat_B"], f"columnas dummy inesperadas: {cols}"
    col = cols[0]
    assert dm.X_train[col].n_unique() == 2

    # val/test solo tienen filas 'B' → un único valor, el del nivel B (el mayor
    # tras el escalado con p_B > 0.5), NO el del nivel de referencia
    b_value = dm.X_train[col].max()
    for name, part in (("val", dm.X_val), ("test", dm.X_test)):
        assert part[col].n_unique() == 1, f"{col} inconsistente en {name}"
        assert part[col].unique().to_list() == pytest.approx([b_value]), (
            f"las filas del segundo nivel quedaron codificadas como el nivel de "
            f"referencia en {name} (bug de to_dummies por frame)"
        )
