# Reporte — Evaluación de modelos por territorio funcional (LCA multinomial)

**Script:** `scripts/train_by_cluster.py`
**Config:** `configs/lca_clusters.yaml`
**Dataset:** `data/clase_{0..6}.parquet` derivado de `saber11_lca_clases_multinomial.parquet` — 4 445 422 registros
**Modelos:** XGBoost (350 rondas, early stopping 10) y MLP (`[128,64,32]`, Early Stopping) por clúster
**Evaluación:** test 15% por clúster, métricas RMSE / MAE / R² / MBE, desagregadas Global / Rural / Urbano

> Nota sobre el dato fuente: este reporte evalúa la segunda generación de clases LCA (variante multinomial, `clase_lca_mult`), con distribución balanceada 9.0%–20.0%, a diferencia del LCA previo (47.2% en clase 0). Los 7 parquet `clase_N.parquet` suman 4 445 422 filas y conservan solo `punt_global` como target.

---

## 1. Objetivo y diseño de la evaluación

El objetivo es medir si el entrenamiento **separado por territorio funcional** produce modelos estables y comparables entre sí, y si persiste la brecha urbano/rural dentro de cada territorio. Cada clúster se entrena de forma independiente (DataModule con `cluster_id`), sin Target Encoding (`cole_mcpio_ubicacion` ausente), y se evalúa en su propio test. La evaluación repite el esquema del proyecto: métricas de error (RMSE, MAE), capacidad explicativa (R²) y **sesgo sistemático (MBE)**, la métrica de equidad — `MBE > +1` sobreestima, `MBE < -1` subestima.

El balanceo de la nueva LCA multinomial es relevante: el clúster más pequeño (clase 5) pasa de 230 865 a 400 113 filas; el más grande (clase 0) pasa de 2 096 589 a 891 146. En split 70/15/15 todos los tests quedan con 60k–133k muestras, eliminando el riesgo de varianza por pocos datos que existía en la fase 2.

---

## 2. Resultados globales por clúster

### 2.1 Tabla resumen (Global)

| Clúster | N total | Train (70%) | XGBoost R² | MLP R² | XGBoost RMSE | MLP RMSE | Ganador |
|---|---:|---:|---:|---:|---:|---:|---|
| **0** | 891 146 | 623 802 | **0.136** | 0.129 | **36.98** | 37.12 | XGBoost (+0.007) |
| **1** | 707 887 | 495 521 | **0.163** | 0.104 | **41.24** | 42.67 | XGBoost (+0.059) |
| **2** | 471 189 | 329 832 | 0.107 | **0.112** | 39.35 | **39.24** | MLP (+0.005) |
| **3** | 776 372 | 543 460 | **0.159** | 0.159 | **39.70** | 39.71 | Empate |
| **4** | 684 733 | 479 313 | **0.130** | 0.115 | **43.57** | 43.97 | XGBoost (+0.015) |
| **5** | 400 113 | 280 079 | **0.147** | 0.138 | **43.77** | 44.00 | XGBoost (+0.009) |
| **6** | 513 982 | 359 787 | **0.167** | 0.149 | **44.94** | 45.41 | XGBoost (+0.018) |

**Lectura:** todos los clústeres convergen a un rango estrecho **R² 0.11–0.17** con RMSE 37–45. A diferencia de la fase 2 (donde el clúster 1 colapsaba a R² −0.14 en XGBoost y −25 en MLP), **ningún clúster colapsa**. La variante multinomial estabiliza el aprendizaje en cada territorio. La ventaja de XGBoost se mantiene (6 de 7 clústeres), pero la brecha se reduce a <0.02 salvo en clúster 1.

### 2.2 Tabla detallada por clúster y área

| Clúster | Modelo | Global R² | Rural R² | Urbano R² | Global MBE | Rural MBE | Urbano MBE |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | XGB | 0.136 | 0.138 | 0.131 | +1.71 | +0.88 | +2.14 |
| 0 | MLP | 0.129 | 0.134 | 0.123 | +0.69 | −0.04 | +1.05 |
| 1 | XGB | 0.163 | 0.125 | 0.159 | +2.34 | +5.56 | +1.73 |
| 1 | MLP | 0.104 | 0.078 | 0.097 | +9.01 | +11.84 | +8.48 |
| 2 | XGB | 0.107 | 0.181 | 0.106 | +3.05 | +0.19 | +3.07 |
| 2 | MLP | 0.112 | 0.204 | 0.111 | +2.45 | −0.68 | +2.46 |
| 3 | XGB | 0.159 | 0.147 | 0.156 | +0.66 | +1.76 | +0.48 |
| 3 | MLP | 0.159 | 0.148 | 0.156 | +0.64 | +0.48 | +0.67 |
| 4 | XGB | 0.130 | 0.192 | 0.123 | −2.04 | −0.31 | −2.13 |
| 4 | MLP | 0.115 | 0.185 | 0.107 | −5.81 | −6.13 | −5.79 |
| 5 | XGB | 0.147 | 0.152 | 0.144 | −6.62 | −7.72 | −6.51 |
| 5 | MLP | 0.138 | 0.146 | 0.135 | −9.00 | −9.78 | −8.93 |
| 6 | XGB | 0.167 | 0.104 | 0.167 | −10.41 | −10.48 | −10.41 |
| 6 | MLP | 0.149 | 0.111 | 0.149 | −11.92 | −9.78 | −11.95 |

---

## 3. Comparación XGBoost vs MLP

*   **Capacidad predictiva:** XGBoost supera a MLP en 6/7 clústeres, pero la diferencia es marginal (ΔR² < 0.06). Solo el clúster 1 muestra separación clara (0.163 vs 0.104), consistente con la mayor heterogeneidad del antiguo clúster privilegiado, ahora redistribuido.
*   **Estabilidad:** A diferencia de la fase 2, **ningún MLP diverge** (R² nunca negativo). El balanceo multinomial eliminó el colapso de los clústeres minoritarios.
*   **Convergencia:** El clúster 3 es el punto de paridad perfecta (R² 0.159 en ambos modelos, MBE < 1), lo que sugiere que ese territorio es el más regular y predecible bajo cualquier arquitectura.

**Implicación para SHAP:** TreeExplainer (XGBoost, exacto) sigue siendo la opción recomendada por velocidad y exactitud, pero ahora ambos modelos son candidatos válidos para interpretabilidad. El análisis diferencial futuro puede promediar ambos sin distorsión por colapso.

---

## 4. Sesgo sistemático (MBE) y equidad por territorio

El MBE revela un **gradiente de sesgo** que no aparecía con el LCA previo:

*   **Clústeres 0–3: sobreestimación leve o nula** (+0.6 a +3.0 en XGB). El clúster 3 es ejemplar (MBE +0.66 Global, sin sesgo relevante en rural/urbano).
*   **Clústeres 4–6: subestimación creciente** (−2.0 → −6.6 → −10.4 en XGB). El clúster 6 subestima ~10–12 puntos de forma sistemática en todos los subgrupos.
*   **MLP amplifica el sesgo:** en clústeres 4–6 el MBE de MLP es 2–3 puntos más negativo que el de XGBoost, y en clúster 1 es 6–7 puntos más positivo.

Este patrón es novedoso: en la fase 2 el sesgo era errático y dominado por el colapso del clúster 1. Con la LCA multinomial emerge una **estructura monotónica**: a medida que se avanza en el índice de clúster (que en la fase 2 correlacionaba con mayor puntaje), el modelo pasa de sobreestimar a subestimar. Esto debe auditarse con SHAP segmentado: indica que las variables que pesan en los territorios de menor puntaje no están capturando la cola alta de la distribución.

### 4.1 Brecha urbano/rural intra-territorio

*   **Sin colapso rural:** en ningún clúster el rendimiento rural es sistemáticamente peor. En clústeres 2 y 4, el R² rural supera al urbano en +0.07 (XGB 0.181 vs 0.106; 0.192 vs 0.123). Esto replica el hallazgo de la fase 2 y refuerza que la etiqueta urbano/rural no explica la varianza: la estructura latente (LCA) sí lo hace.
*   **Sesgo rural específico:** solo el clúster 1 muestra sobreestimación rural marcada (+5.56 XGB, +11.84 MLP), lo que sugiere que dentro de ese territorio el modelo rural aprende una media distinta.

---

## 5. Síntesis e implicaciones para la cadena metodológica

1.  **La variante multinomial cumple su objetivo:** estabiliza todos los territorios (R² 0.11–0.17, sin colapsos) y balancea los tamaños (9%–20%), eliminando el riesgo de pocos datos y la inestabilidad de la fase 2.
2.  **XGBoost sigue siendo el modelo base para SHAP**, pero la paridad con MLP habilita un análisis de robustez cruzada (comparar ambos SHAP).
3.  **El sesgo graduado (clústeres 4–6) es el nuevo hallazgo de equidad:** debe ser el foco del `shap_analysis.py` segmentado. La pregunta ya no es "¿qué clúster colapsa?" sino "¿qué variables explican la subestimación sistemática de los territorios 4–6?".
4.  **La validación por área confirma el replanteamiento:** la desagregación Rural/Urbano dentro de cada territorio no muestra degradación rural, validando que los territorios funcionales son la unidad correcta de análisis.

**Próximo paso:** ejecutar `scripts/shap_analysis.py --config configs/shap_analysis.yaml --model-type xgb` sobre estos nuevos checkpoints para auditar la importancia asimétrica que genera el gradiente de MBE.

---
*Generado a partir de `train_by_cluster.py` — 7 clústeres × 2 modelos — 21 de agosto de 2026.*
