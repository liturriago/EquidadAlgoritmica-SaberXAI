# EquidadAlgoritmica-SaberXAI

Predicción del `punt_global` del ICFES Saber 11 con **XGBoost** y **PyTorch MLP**,
con análisis de equidad por territorio funcional (clústeres LCA) mediante **MBE**
(Mean Bias Error) y **SHAP/XAI**.

## Estructura

```
configs/          # YAML de experimentos (default, lca_clusters, global_baseline, shap_analysis)
src/saber_xai/    # Paquete: config, data (DataModule), models (XGB, MLP, evaluator)
scripts/          # train.py (global/cluster), train_by_cluster.py, diagnostico_mbe.py,
                  # shap_analysis.py
notebooks/        # 01-08 exploración/clustering · 09-12 experimentos de Kaggle
tests/            # pytest (27 tests)
documentacion/    # fase_1..fase_4 (guía de repetición del entrenamiento en fase_4)
```

## Entrenamiento (resumen)

```bash
uv sync --extra dev

# Línea base global sobre la unión de clústeres (mismos splits que los modelos por clúster)
uv run python scripts/train.py --config configs/global_baseline.yaml --mode global
uv run python scripts/train.py --config configs/global_baseline.yaml --mode global_indicators

# Un modelo por clúster LCA (0-6)
uv run python scripts/train.py --config configs/lca_clusters.yaml --mode cluster

# Prueba decisiva del MBE (hipótesis H1 vs H5)
uv run python scripts/diagnostico_mbe.py --config configs/lca_clusters.yaml --model-type both
```

En Kaggle: usar los notebooks `09`–`12` (clonan este repo e instalan con
`pip install -e .`), adjuntando el dataset `lucasiturriago/pruebassaber11`.
Guía completa con salidas esperadas y troubleshooting:
[`documentacion/fase_4/guia_repetir_entrenamiento.md`](documentacion/fase_4/guia_repetir_entrenamiento.md).

## Tests

```bash
uv run pytest tests/ -q
```

## Notas de reproducibilidad

- Semilla única `random_state: 42`; split 70/15/15 replicado por clúster.
- XGBoost guarda sidecar `.meta.json` con `feature_names` y `best_iteration`.
- El MLP guarda los mejores pesos del early stopping en `models/mlp_*.pth`.
- La carga de datos es estricta: sin parquet no hay datos dummy (solo con
  `--allow-dummy`, para tests).
