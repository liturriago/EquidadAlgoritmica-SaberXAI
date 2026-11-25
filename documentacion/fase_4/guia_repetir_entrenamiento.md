# Guía — Repetición del entrenamiento (XGBoost + MLP) y línea base global

> Fase A2 del Entregable 3. Cierra: (1) línea de base global por territorio
> (comentario contenido 1, parte 1) y (2) el diagnóstico del MBE −10,41
> (comentario contenido 1, punto 2).
>
> Actualizado: 2026-11-25 · Repo: `EquidadAlgoritmica-SaberXAI` rama `main`.

## 1. Qué corre esta guía

| Experimento | Script | Modo | Notebook Kaggle |
|---|---|---|---|
| Modelos por clúster LCA (0–6) | `scripts/train.py` | `--mode cluster` | `09_entrenamiento_modelos.ipynb` |
| SHAP / XAI por clúster | `scripts/shap_analysis.py` | — | `10_shap_XAI.ipynb` |
| **Línea base global** (sin indicadoras) | `scripts/train.py` | `--mode global` | `11_linea_base_global.ipynb` |
| **Global + indicadoras de territorio** | `scripts/train.py` | `--mode global_indicators` | `11_linea_base_global.ipynb` |
| **Diagnóstico MBE (H1 vs H5)** | `scripts/diagnostico_mbe.py` | — | `12_diagnostico_mbe.ipynb` |

**Propiedad científica clave:** el modo `global` replica el split 70/15/15 **por
clúster** (shuffle semilla 42) y luego une los conjuntos, de modo que el modelo
global se evalúa sobre **exactamente las mismas filas de test** que los modelos
por clúster. La comparación de R²/RMSE/MBE por territorio (Tabla 4) es así
válida. No "arreglar" este orden: cambiarlo invalidaría la comparación.

## 2. Correr en Kaggle (recomendado)

1. Abrir el notebook (`09`–`12`) en Kaggle y adjuntar el dataset
   `lucasiturriago/pruebassaber11` (Add Data → Datasets). **Sin el dataset,
   los scripts fallan a propósito** (la carga es estricta; el fallback a datos
   dummy solo existe para tests).
2. Elegir GPU (Settings → Accelerator → GPU) para el MLP; XGBoost corre en CPU.
3. Correr todas las celdas. La primera clona
   `https://github.com/liturriago/EquidadAlgoritmica-SaberXAI.git` y hace
   `pip install -e .` — **los cambios deben estar en GitHub** para que Kaggle
   los vea.
4. Recuperar de `/kaggle/working/` (Output del notebook):
   - `results/tabla4.csv` — filas Global / Área / Clúster (insumo de la Tabla 4).
   - `results/metrics_*.json` — métricas completas por ámbito.
   - `results/diagnostico_mbe.json` — veredicto H1/H5 por clúster.
   - `results/split_meta_*.json` — tamaños y medias del split (trazabilidad).
   - `results/mlp_history_*.json` — curvas de entrenamiento (figuras).
   - `models/*.json|.pth` + sidecars `.meta.json` (features y best_iteration).

Cuotas: la cola de GPU de Kaggle da 30 h/semana (verificación: herramienta MCP
`get_accelerator_quota`).

### Lanzamiento remoto vía MCP de Kaggle (desde OpenCode)

El servidor MCP oficial de Kaggle está configurado en este equipo
(`~/.config/opencode/opencode.json`, token KGAT en `~/.kaggle/access_token`).
Flujo: `save_notebook` (sube y ejecuta de arriba a abajo) →
`get_notebook_session_status` (seguido) → `download_notebook_output_zip`
(resultados). Los notebooks deben llevar `datasetDataSources:
["lucasiturriago/pruebassaber11"]` y GPU habilitada.

## 3. Correr en local

```bash
cd EquidadAlgoritmica-SaberXAI
uv sync --extra dev          # crea .venv con todas las dependencias

# Poner los parquet en una carpeta (p. ej. data/) y ajustar data_dir en el YAML,
# o sobreescribir con un config local.

# Línea base global (± indicadoras)
uv run python scripts/train.py --config configs/global_baseline.yaml --mode global
uv run python scripts/train.py --config configs/global_baseline.yaml --mode global_indicators

# Modelos por clúster (compat. con shap_analysis: xgb_cluster_{id}.json)
uv run python scripts/train.py --config configs/lca_clusters.yaml --mode cluster

# Diagnóstico del MBE (requiere modelos por clúster guardados en models/)
uv run python scripts/diagnostico_mbe.py --config configs/lca_clusters.yaml --model-type both

# Tests (27)
uv run pytest tests/ -q
```

`scripts/train_by_cluster.py` sigue funcionando con su CLI histórica
(`--config --clusters --models-dir`) como wrapper de `train.py --mode cluster`.

## 4. Salidas y su destino en el artículo

| Salida | Uso |
|---|---|
| `results/tabla4.csv` | **Tabla 4** (R²/RMSE/MBE por territorio, global vs por clúster) |
| `metrics_global*.json` → `por_cluster` | Ajuste de la afirmación del resumen/discusión a la comparación |
| `results/diagnostico_mbe.json` | §5.2 + nota de la Figura 5 (cierre del MBE −10,41) |
| `models/*.meta.json` (`feature_names`) | Corrige la discrepancia "60–68 vs 58–75" features en §5.2 |
| `results/mlp_history_*.json` | Figuras de convergencia (Anexo: parada temprana) |

## 5. Reproducibilidad y garantías

- Semilla única `random_state: 42` (numpy + torch + shuffle de polars).
- Split idéntico al histórico (barajado → test 15 % / val 15 % / train 70 %) para
  mantener comparabilidad con los modelos ya entrenados.
- XGBoost predice con `best_iteration` (sidecar `.meta.json`); el MLP guarda y
  restaura **los mejores pesos** del early stopping (`mlp_*.pth`).
- Los conteos de features por modelo quedan en los sidecars → cualquier
  discrepancia futura es verificable por corrida.

## 6. Problemas conocidos

| Síntoma | Causa / solución |
|---|---|
| `FileNotFoundError` al iniciar | Dataset no adjunto o `data_dir` incorrecto. Es intencional: no se aceptan datos dummy en ejecuciones reales. |
| Kaggle clona código viejo | El cambio no está en GitHub: hacer push antes de correr el notebook. |
| MBE del MLP enorme y R² muy negativo | Pesos obsoletos de otra corrida en `models/` (problema ya corregido en `mlp_model.py`; regenerar modelos). |
| `Invalid auth scope: resources.admin:` en OAuth de Kaggle | Incompatibilidad conocida de OpenCode con los scopes de Kaggle. Usar token KGAT (`~/.kaggle/access_token` + header en `opencode.json`), que es el método configurado en este equipo. |
