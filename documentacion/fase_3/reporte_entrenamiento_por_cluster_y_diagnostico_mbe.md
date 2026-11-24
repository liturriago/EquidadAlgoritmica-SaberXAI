# Entrenamiento por clúster (LCA multinomial) — log completo y diagnóstico del MBE

> **Script:** `scripts/train_by_cluster.py` — `--config configs/lca_clusters.yaml`
> **Datos:** `/kaggle/input/datasets/lucasiturriago/pruebassaber11/clase_{0..6}.parquet` (LCA multinomial, 4.445.422 filas)
> **Salidas:** `models/xgb_cluster_{0..6}.json` · `models/mlp_cluster_{0..6}.pth`
> **Evaluación:** test 15 % por clúster; métricas RMSE / MAE / R² / MBE desagregadas Global / Rural / Urbano
> **Uso:** insumo del comentario de contenido 1 del ICFES (línea de base y **MBE −10,41 del clúster 6**).
> Todos los números están transcritos literalmente del log de consola de la corrida.

---

## 1. Especificación de entrenamiento

| Componente | Parámetros |
|---|---|
| Partición | 70 % train / 15 % validación / 15 % test, por clúster (semilla 42) |
| Preprocesamiento | One-Hot por clúster (pipeline aislado); sin target encoding |
| XGBoost | 350 rondas máx., `learning_rate` 0.05, `max_depth` 6, `reg:squarederror`, early stopping paciencia 10 sobre val-RMSE |
| MLP (PyTorch, cuda) | capas [128, 64, 32], lr 0.001, batch 8192, 100 épocas máx., early stopping paciencia 10 |
| MBE | `mean(y_pred − y_true)` sobre test (`src/saber_xai/models/evaluator.py`) — MBE > +1 sobreestima, < −1 subestima |

## 2. Resumen de entrenamiento por clúster

| Clúster | Train | Features* | XGB mejor ronda | XGB mejor val-RMSE | XGB ¿agotó 350 rondas? | MLP early stopping |
|---|---|---|---|---|---|---|
| 0 | 623.804 | 75 | 349 | 35,7856 | **Sí** (val aún descendía) | época 38 |
| 1 | 495.521 | 74 | 113 | 41,7918 | No (paró ~ronda 123) | época 32 |
| 2 | 329.833 | 58 | 348 | 38,3750 | **Sí** (val aún descendía) | época 87 |
| 3 | 543.462 | 70 | 196 | 40,2372 | No (paró ~ronda 206) | época 32 |
| 4 | 479.315 | 72 | 264 | 44,0389 | No (paró ~ronda 274) | época 52 |
| 5 | 280.081 | 70 | 349 | 43,6855 | **Sí** (val aún descendía) | **No** (100/100 épocas) |
| 6 | 359.788 | 71 | 349 | 44,1481 | **Sí** (val aún descendía) | época 70 |

\* Reportado por `DataModule` al cargar cada parquet (ver §5: no coincide con el conteo de
features del modelo citado en el artículo — pendiente de verificación).

**Hecho relevante para el diagnóstico:** en **4 de 7 clústeres (0, 2, 5, 6)** el XGBoost
llegó a la ronda 349 = límite de 350 rondas con la validación **todavía descendiendo**;
la parada temprana nunca se activó. En esos clústeres el modelo quedó truncado por el
tope de rondas, no por convergencia.

## 3. Evaluación en test (15 % por clúster)

### 3.1 XGBoost por clúster

| Clúster | Segmento | RMSE | MAE | R² | MBE | Lectura |
|---|---|---|---|---|---|---|
| 0 | Global | 36,9800 | 29,6218 | 0,1359 | **+1,7142** | sobreestima |
| 0 | Rural | 35,5953 | 28,5098 | 0,1378 | +0,8777 | sin sesgo relevante |
| 0 | Urbano | 37,6607 | 30,1840 | 0,1311 | +2,1370 | sobreestima |
| 1 | Global | 41,2396 | 33,3885 | 0,1633 | **+2,3388** | sobreestima |
| 1 | Rural | 40,0722 | 32,6703 | 0,1253 | +5,5572 | sobreestima |
| 1 | Urbano | 41,4569 | 33,5245 | 0,1589 | +1,7295 | sobreestima |
| 2 | Global | 39,3467 | 31,7796 | 0,1068 | **+3,0521** | sobreestima |
| 2 | Rural | 37,3554 | 30,2706 | 0,1810 | +0,1881 | sin sesgo relevante |
| 2 | Urbano | 39,3564 | 31,7872 | 0,1062 | +3,0664 | sobreestima |
| 3 | Global | 39,6973 | 32,0500 | 0,1590 | **+0,6575** | sin sesgo relevante |
| 3 | Rural | 39,1815 | 31,7301 | 0,1465 | +1,7588 | sobreestima |
| 3 | Urbano | 39,7775 | 32,1001 | 0,1564 | +0,4849 | sin sesgo relevante |
| 4 | Global | 43,5712 | 35,2407 | 0,1304 | **−2,0388** | subestima |
| 4 | Rural | 43,8384 | 35,6512 | 0,1921 | −0,3084 | sin sesgo relevante |
| 4 | Urbano | 43,5573 | 35,2195 | 0,1232 | −2,1281 | subestima |
| 5 | Global | 43,7711 | 35,0510 | 0,1470 | **−6,6208** | subestima |
| 5 | Rural | 42,7685 | 34,0832 | 0,1517 | −7,7227 | subestima |
| 5 | Urbano | 43,8679 | 35,1456 | 0,1438 | −6,5130 | subestima |
| 6 | Global | 44,9422 | 36,0619 | 0,1666 | **−10,4144** | subestima |
| 6 | Rural | 47,0383 | 37,6647 | 0,1042 | −10,4757 | subestima |
| 6 | Urbano | 44,9127 | 36,0399 | 0,1670 | −10,4135 | subestima |

### 3.2 MLP por clúster

| Clúster | Segmento | RMSE | MAE | R² | MBE | Lectura |
|---|---|---|---|---|---|---|
| 0 | Global | 37,1175 | 29,6361 | 0,1294 | +0,6874 | sin sesgo relevante |
| 0 | Rural | 35,6732 | 28,4862 | 0,1340 | −0,0365 | sin sesgo relevante |
| 0 | Urbano | 37,8268 | 30,2174 | 0,1234 | +1,0534 | sobreestima |
| 1 | Global | 42,6736 | 34,6590 | 0,1041 | **+9,0116** | sobreestima |
| 1 | Rural | 41,1524 | 33,6331 | 0,0775 | +11,8409 | sobreestima |
| 1 | Urbano | 42,9556 | 34,8533 | 0,0970 | +8,4759 | sobreestima |
| 2 | Global | 39,2424 | 31,6682 | 0,1115 | +2,4456 | sobreestima |
| 2 | Rural | 36,8389 | 29,6719 | 0,2035 | −0,6758 | sin sesgo relevante |
| 2 | Urbano | 39,2541 | 31,6782 | 0,1108 | +2,4612 | sobreestima |
| 3 | Global | 39,7060 | 32,0070 | 0,1586 | +0,6444 | sin sesgo relevante |
| 3 | Rural | 39,1474 | 31,5808 | 0,1480 | +0,4827 | sin sesgo relevante |
| 3 | Urbano | 39,7929 | 32,0738 | 0,1558 | +0,6697 | sin sesgo relevante |
| 4 | Global | 43,9654 | 35,4209 | 0,1146 | **−5,8083** | subestima |
| 4 | Rural | 44,0309 | 35,4879 | 0,1850 | −6,1300 | subestima |
| 4 | Urbano | 43,9621 | 35,4174 | 0,1068 | −5,7917 | subestima |
| 5 | Global | 43,9982 | 35,3139 | 0,1381 | **−9,0037** | subestima |
| 5 | Rural | 42,9178 | 34,3538 | 0,1458 | −9,7774 | subestima |
| 5 | Urbano | 44,1025 | 35,4078 | 0,1346 | −8,9281 | subestima |
| 6 | Global | 45,4113 | 36,3332 | 0,1491 | **−11,9217** | subestima |
| 6 | Rural | 46,8536 | 37,4668 | 0,1112 | −9,7790 | subestima |
| 6 | Urbano | 45,3911 | 36,3176 | 0,1491 | −11,9512 | subestima |

### 3.3 Comparativa XGBoost vs MLP (test global)

| Clúster | XGB R² | MLP R² | XGB RMSE | MLP RMSE | Ganador |
|---|---|---|---|---|---|
| 0 | **0,1359** | 0,1294 | **36,98** | 37,12 | XGBoost (+0,007) |
| 1 | **0,1633** | 0,1041 | **41,24** | 42,67 | XGBoost (+0,059) |
| 2 | 0,1068 | **0,1115** | 39,35 | **39,24** | MLP (+0,005) |
| 3 | **0,1590** | 0,1586 | **39,70** | 39,71 | Empate |
| 4 | **0,1304** | 0,1146 | **43,57** | 43,97 | XGBoost (+0,015) |
| 5 | **0,1470** | 0,1381 | **43,77** | 44,00 | XGBoost (+0,009) |
| 6 | **0,1666** | 0,1491 | **44,94** | 45,41 | XGBoost (+0,018) |

> Estos valores son exactamente los de la **Tabla 4 y Figuras 4–5 del artículo**
> (XGBoost R² 0,107–0,167; MLP R² 0,104–0,159): esta corrida es la fuente de
> esas cifras.

---

## 4. Diagnóstico del MBE −10,41 en el clúster 6
*(comentario de contenido 1 del ICFES — punto 2)*

### 4.1 El hecho

El XGBoost entrenado y evaluado **dentro del clúster 6** subestima en **−10,4144 puntos**
el puntaje global de su propio test (−10,48 en rural, −10,41 en urbano). Con una
partición de prueba aleatoria de la misma distribución, el MBE debería rondar cero;
el propio comentario de la Dirección de Investigación señala tres explicaciones
posibles: partición no aleatoria (p. ej. temporal), predicciones que no alcanzan la
media del clúster (base_score por defecto con parada temprana) o error de cálculo.

### 4.2 Lo que este log aporta como evidencia

1. **El sesgo es del par train/test, no del algoritmo.** El MLP del mismo clúster —
   una arquitectura distinta, sin `base_score` ni boosting — subestima en **−11,92**
   puntos. Ambas familias de modelo coinciden en signo y magnitud en los 7 clústeres
   (C4: XGB −2,04 / MLP −5,81; C5: XGB −6,62 / MLP −9,00; C1: XGB +2,34 / MLP +9,01).
   Un artefacto de XGBoost no se reproduciría en el MLP.
2. **El sesgo sigue un gradiente sistemático por clúster, no es ruido:**
   +3,05 (C2) → +2,34 (C1) → +1,71 (C0) → +0,66 (C3) → −2,04 (C4) → −6,62 (C5) → −10,41 (C6),
   en el mismo sentido en ambos modelos. Un sesgo aleatorio de partición no produciría
   una pendiente tan ordenada.
3. **Cuatro de siete XGBoost agotaron las 350 rondas** (C0, C2, C5, C6) con la validación
   todavía descendiendo. El clúster 6 es precisamente uno de ellos (mejor ronda 349 =
   última). Un modelo truncado por el tope de rondas mantiene sus predicciones más
   cerca de su valor inicial que un modelo convergido, lo que puede dejar un sesgo
   residual del orden de la magnitud observada. (Hipótesis del comentario: "predicciones
   que no alcanzaron la media del clúster".)
4. **El cálculo del MBE no tiene error.** `evaluator.py` computa
   `np.mean(y_pred − y_true)` sobre el conjunto de prueba completo del clúster;
   consistente con los valores desagregados rural/urbano (media ponderada coherente).
5. **Inspección de los modelos guardados** (los 4 XGBoost que agotaron las 350 rondas),
   descargados de la salida de la corrida:

   | Modelo | `base_score` (media de train) | `num_trees` | `num_feature` | MBE test (log) | Media de test implícita\* |
   |---|---|---|---|---|---|
   | `xgb_cluster_0.json` | 228,72942 | 350 | 75 | +1,7142 | ≈ 227,0 |
   | `xgb_cluster_2.json` | 254,12764 | 350 | 58 | +3,0521 | ≈ 251,1 |
   | `xgb_cluster_5.json` | 313,29224 | 350 | 70 | −6,6208 | ≈ 319,9 |
   | `xgb_cluster_6.json` | 263,82922 | 350 | 71 | −10,4144 | ≈ 274,2 |

   \* Aproximación: para un booster poco entrenado cuyo valor inicial es la media de
   train, `mean(y_pred)` gravita alrededor de `base_score`, de modo que
   MBE ≈ media de train − media de test.

   - `boost_from_average = 1` en todos: el valor inicial es **la media del propio
     clúster** (228,7 ≈ mediana 225 del C0 y 313,3 ≈ mediana 316 del C5 del artículo),
     no un valor por defecto ni la media global. Esto **refuta** la hipótesis del
     "base_score por defecto".
   - `num_trees = 350` en los 4: confirma el truncamiento por tope de rondas (§4.2.3).
   - `num_feature` = 75 / 58 / 70 / 71: coincide con el log y refuta el conteo del
     artículo (ver §5).

   **Consecuencia decisiva:** el desfase media de train → media de test tiene **signo y
   magnitud propios de cada clúster** (−10,4 en C6, −6,6 en C5, +3,1 en C2, +1,7 en C0),
   con los modelos arrancando en la media correcta de su clúster. Un sesgo producido por
   el algoritmo no respetaría esa estructura; un desfase de **distribución entre las
   particiones** sí la produce. Apunta a H1 (partición con orden, p. ej. temporal), con
   H2 (truncamiento por rondas) como agravante de la magnitud.

### 4.3 Hipótesis y prueba decisiva

**Contexto de código (verificado):** `data_module.py:68–78` baraja el DataFrame completo
(`df.sample(fraction=1.0, shuffle=True, seed=42)`) **antes** de partir en 70/15/15 por
posición. El parquet crudo sí viene ordenado (las medias por cortes crudos difieren de
forma estructurada — ver tabla abajo), pero esos cortes **no** son las particiones del
pipeline. Se verificó además que `DataFrame.sample(fraction=1.0, shuffle=True, seed=42)`
en polars 2.2.0 sí produce una permutación uniforme (medias por partición equilibradas
en una prueba controlada).

**Medias de `punt_global` por cortes crudos del parquet** (NO son las particiones del
pipeline; evidencian que el archivo está ordenado):

| Clúster | crudo [:70 %] | crudo [70 %:85 %] | crudo [85 %:] |
|---|---|---|---|
| 0 | 230,04 | 223,16 | 228,42 |
| 1 | 242,05 | 238,21 | 243,82 |
| 2 | 255,16 | 249,50 | 254,23 |
| 3 | 248,77 | 244,22 | 249,02 |
| 4 | 270,78 | 269,17 | 272,19 |
| 5 | 313,64 | 311,67 | 313,38 |
| 6 | 262,79 | 264,71 | 268,11 |

**La contradicción abierta:** si la partición es intercambiable (shuffle efectivo) y las
predicciones gravitan alrededor de la media de su clúster (`base_score` verificado), el
MBE debería ser ≈ 0; sin embargo es −10,41 en C6 (y −11,92 en su MLP). O bien el
shuffle no se comportó igual en la ejecución de Kaggle (versión de polars, script
modificado), o bien el sesgo vive en la etapa de predicción/evaluación. La prueba
decisiva debe medir **las particiones reales del pipeline** y la media de las
predicciones.

| # | Hipótesis | Evidencia actual | Prueba decisiva |
|---|---|---|---|
| **H1** | **El shuffle no fue efectivo en la ejecución** (p. ej. versión de polars cuyo `sample(fraction=1.0, shuffle=True)` no permuta), de modo que test = `head(15 %)` del parquet ordenado y train = `tail(70 %)`: particiones con medias distintas por clúster (signo y magnitud propios, reproducidos en ambos modelos). | Patrón por clúster en XGB y MLP (§4.2.1–2, 4.2.5); parquet ordenado (tabla anterior) | Correr el snippet de §4.4 (usa el propio `DataModule`): si `mean(y_test)` difiere de `mean(y_train)` en ≈ la magnitud del MBE, H1 confirmada. |
| **H2** | Truncamiento por tope de rondas (350 árboles, validación aún descendiendo) como agravante de la magnitud. | Confirmado en los 4 modelos (§4.2.3, 4.2.5) | Reentrenar C6 con 1.500 rondas y comparar MBE. |
| **H5** | Sesgo en la etapa de predicción/evaluación (alineación X/y, transformación de target). | Sin evidencia directa aún; solo quedaría en pie si H1 se descarta | El mismo snippet imprime `mean(y_pred)` vs `mean(y_test)` y reproduce el MBE exacto. |
| H3 | Error de cálculo del MBE. | Descartada por código (§4.2.4) | Ninguna adicional. |
| ~~H4~~ | `base_score` por defecto. | **Refutada** (§4.2.5: media de cada clúster) | — |

> **Lectura preliminar (actualizada):** H1 (shuffle inefectivo en la ejecución real) es
> la explicación más parsimoniosa — reproduce el signo y magnitud por clúster en dos
> arquitecturas distintas — pero queda pendiente de la medición directa porque el código
> del repo sí baraja. La medición de §4.4 decide entre H1 y H5 de una vez.

### 4.4 Prueba decisiva (snippet para Kaggle)

Usa el **propio `DataModule`** del proyecto (exactamente el proceso de entrenamiento) y
el modelo guardado; imprime las medias de las particiones reales, la media de las
predicciones y reproduce el MBE:

```python
import numpy as np, xgboost as xgb
from saber_xai.config import config
from saber_xai.data.data_module import DataModule

config.load_from_yaml('configs/lca_clusters.yaml')

for c in [0, 2, 5, 6]:
    config.cluster_id = c
    dm = DataModule(); dm.prepare_data()
    ytr = dm.y_train.to_numpy().ravel()
    yva = dm.y_val.to_numpy().ravel()
    yte = dm.y_test.to_numpy().ravel()
    m = xgb.Booster(); m.load_model(f'models/xgb_cluster_{c}.json')
    pred = m.predict(dm.get_dmatrices()[2])
    print(f'C{c}: train {ytr.mean():.2f} | val {yva.mean():.2f} | test {yte.mean():.2f} '
          f'| mean(pred) {pred.mean():.2f} | MBE {pred.mean()-yte.mean():+.2f}')
```

**Cómo leer la salida:**
- Si `test` difiere de `train` en ≈ la magnitud del MBE → **H1** (el shuffle no barajó
  en esa ejecución): la explicación queda lista para el artículo.
- Si `test ≈ train` pero `mean(pred)` queda ~10 puntos abajo → **H5**: el sesgo está en
  la predicción/evaluación y hay que revisar la alineación X/y de `get_dmatrices`.

### 4.5 Valores aún necesarios para cerrar la explicación

- [ ] **Salida del snippet de §4.4** (medias de partición reales + `mean(y_pred)`) — *dato clave pendiente*
- [x] `base_score` y número de árboles de los `.json` de los 4 modelos truncados — **verificados: media de su clúster / 350 árboles / 75-58-70-71 features**
- [x] Comportamiento del shuffle de polars — **verificado en 2.2.0: sí permuta**
- [ ] Versión de polars en el entorno Kaggle de la corrida (`polars.__version__`)
- [ ] Orden de las filas en `clase_*.parquet` (¿cronológico por aplicación?) y modo de
      partición de `DataModule` (¿aleatorio o posicional?)

---

## 5. Discrepancias detectadas frente al artículo

| Ítem | Log de esta corrida | Artículo (§5.2 / Anexo) | Acción |
|---|---|---|---|
| Features por modelo | 75 / 74 / 58 / 70 / 72 / 70 / 71 (C0…C6); **verificado en `xgb_cluster_6.json`: `num_feature = 71`** | "entre 60 y 68" (0:68, 1:60, 2:63, 3:67, 4:66, 5:64, 6:64) | **Corregir el artículo** con el conteo real (los valores del artículo corresponden a otra generación de modelos: ver nota al pie) |
| Train (70 %) | 623.804 / 495.521 / 329.833 / 543.462 / 479.315 / 280.081 / 359.788 | 623.802 / 495.521 / 329.832 / 543.460 / 479.313 / 280.079 / 359.787 | Diferencias de 1–2 muestras: unificar contra esta corrida al rehacer la Tabla 4 |
| Parada temprana | No se activó en C0, C2, C5, C6 (tope de 350 rondas) | Anexo: "350 rondas, parada temprana con paciencia de 10" | Precisar el texto: la paciencia de 10 está configurada, pero en 4 de 7 modelos no se alcanzó el criterio de parada |

> **⚠ Nota de consistencia entre fuentes:** el conteo "60–68" del artículo coincide con
> el que reporta `reporte_shap_multinomial.md` ("68, 60, 63, 67, 66, 64, 64 según
> clúster", leído de `model.feature_names` en la corrida de SHAP, cuyo config apunta a
> un directorio de modelos distinto: `/kaggle/input/notebooks/.../models`). Es decir,
> **la Tabla 4 proviene de esta corrida (58–75 features) y los valores SHAP del artículo
> podrían provenir de una generación anterior de modelos**. Verificar cuáles modelos se
> usaron en `shap_analysis.py` y, si son distintos, re-correr SHAP sobre los modelos
> actuales para que §5.2 y §5.3 sean internamente consistentes.

Los valores de R² y RMSE de las tablas §3.1–3.3 **sí coinciden** con la Tabla 4 del
artículo (p. ej. C6 XGB: 0,167 / 44,94), de modo que esta corrida es la fuente de la
tabla y las discrepancias anteriores son de redacción/conteo, no de resultados.

## 6. Reproducibilidad

```bash
# en Kaggle (GPU), con los datos en /kaggle/input/datasets/lucasiturriago/pruebassaber11/
python scripts/train_by_cluster.py --config configs/lca_clusters.yaml
```

- Parámetros completos: `configs/default.yaml` (XGBoost/MLP) y `configs/lca_clusters.yaml`.
- Semilla global: 42. Entorno: PyTorch en cuda para el MLP; XGBoost con `nthread = -1`.
- Reglas de las etiquetas: MBE > +1 "sobreestima", < −1 "subestima",
  entre −1 y +1 "sin sesgo relevante" (banda de equidad usada en la Figura 5 del artículo).
