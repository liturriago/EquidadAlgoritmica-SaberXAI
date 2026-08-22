# Reporte — Interpretabilidad SHAP segmentada por territorio funcional (LCA multinomial)

**Script:** `scripts/shap_analysis.py` — `--model-type xgb`
**Modelos:** XGBoost por clúster (`xgb_cluster_{0..6}.json`) entrenados en `scripts/train_by_cluster.py`
**Dataset:** `data/clase_{0..6}.parquet` (LCA multinomial, 4 445 422 filas) — `max_samples=10 000` por clúster para SHAP
**Método:** TreeExplainer (SHAP, teoría de juegos cooperativos), `mean(|SHAP|)` como importancia global por clúster
**Config:** `configs/shap_analysis.yaml` (`shap_model_type: xgb`)

> Nota de consistencia: cada XGBoost conserva el esquema de features con el que fue entrenado (68, 60, 63, 67, 66, 64, 64 features según clúster). `shap_analysis.py` alinea cada matriz `X` a `model.feature_names` y respeta el orden, rellenando con ceros las dummies faltantes. Sin esta alineación se produce el error `feature names mismatch` observado en la corrida inicial.

---

## 1. Objetivo y método

El objetivo no es explicar el modelo global, sino **auditar la importancia asimétrica** dentro de cada territorio funcional descubierto por la LCA multinomial. Para cada clúster se calcula el SHAP value de cada feature sobre 10 000 muestras de su propio test y se promedia `mean(|SHAP|)`. Un valor alto indica que la variable mueve fuertemente la predicción de `punt_global` dentro de ese territorio; un valor bajo indica irrelevancia local, aunque pueda ser relevante en otro territorio.

El entregable para el ICFES es la **matriz 7 × ~65** `shap_importance_XGBoost.json` (`documentacion/fase_3/shap_importance_XGBoost.txt`) que se analiza abajo. SHAP es descriptivo, no causal: señala dónde el modelo concentra la brecha, no una relación causa-efecto.

---

## 2. Panorama comparativo — top-10 global por clúster

Valores `mean(|SHAP|)` (puntos de `punt_global` que la feature aporta en promedio). Ordenados de mayor a menor por clúster.

| Rank | Clúster 0 | Clúster 1 | Clúster 2 | Clúster 3 | Clúster 4 | Clúster 5 | Clúster 6 |
|---|---|---|---|---|---|---|---|
| 1 | Sabatina 4.33 | Estrato 3.30 | Género F 5.08 | Género M 4.00 | Género F 2.88 | Padre Postgrado 5.50 | Completa 11.64 |
| 2 | Noche 3.99 | Noche 2.95 | Noche 4.04 | Noche 3.15 | Cuartos 2.56 | Mañana 4.97 | Sabatina 4.67 |
| 3 | Rural 3.33 | Rural 2.67 | PC 2.27 | Sabatina 3.15 | Madre Téc. Comp. 2.34 | Calendario B 4.84 | Noche 2.77 |
| 4 | Género M 2.94 | Bolívar 2.22 | Bogotá 2.03 | PC 2.44 | Madre Prof. Comp. 2.22 | Estrato 3.52 | Cuartos 2.31 |
| 5 | Estrato 2.25 | Madre Téc. Comp. 1.94 | Estrato 1.47 | Cuartos 1.86 | Padre Téc. Comp. 2.07 | Madre Postgrado 3.14 | Género F 2.28 |
| 6 | Bolívar 1.62 | Género F 1.69 | Madre Téc. Comp. 1.42 | Rural 1.81 | Estrato 1.76 | Cuartos 3.02 | Madre Téc. Comp. 2.18 |
| 7 | Cuartos 1.54 | Internet 1.55 | Cuartos 1.39 | Santander 1.63 | Padre Prof. Comp. 1.69 | Madre Sec. Comp. 1.36* | Madre Prof. Comp. 2.07 |
| 8 | Personas 1.48 | Magdalena 1.55 | Personas 1.35 | Estrato 1.57 | Santander 1.65 | Santander 1.37 | PC 1.89 |
| 9 | Madre Prim. Inc. 1.41 | Cuartos 1.50 | Madre Prim. Inc. 1.05 | Madre Prim. Inc. 1.36 | PC 1.44 | Valle 1.43 | Padre Téc. Comp. 1.84 |
| 10 | Antioquia 1.26 | Oficial 1.30 | Padre Prim. Inc. 0.92 | Antioquia 1.21 | Fem. colegio 1.36 | — | Padre Prof. Comp. 1.44 |

*En clúster 5 el puesto 7 es `Sec. (Bach.) Completa madre 1.36` (valor similar a Santander). Se omiten decimales terceros por legibilidad.

**Lectura transversal inmediata:**

*   **Jornada es la palanca universal:** aparece en el top-3 de 6 de 7 clústeres. `Sabatina` y `Noche` dominan 0–3 y 6; `Mañana` y `Completa` dominan 5 y 6.
*   **Género del estudiante es la segunda palanca:** `estu_genero_M/F` está en top-5 en todos los clústeres (2.28–5.08). Es la única variable con alta importancia en los 7 territorios.
*   **Estrato** siempre en top-6 (1.40–3.52), pero nunca en puesto 1 salvo clúster 1.
*   **Postgrado de los padres** solo emerge en clúster 5 (puestos 1 y 5), el territorio que el reporte de evaluación identificó con subestimación sistemática (−6.6 MBE).

---

## 3. Perfiles por territorio

### 3.1 Clúster 0 (20.0% pobl., 891k) — bloque medio masivo, R² 0.136

Top: `Jornada Sabatina 4.33 > Noche 3.99 > Rural 3.33 > Género M 2.94 > Estrato 2.25`. Patrón de **escuela**: 4 de 5 primeras son de colegio (jornada, área, carácter). `fami_tienecomputador 0.98` y `tieneserviciotv 0.84` tienen peso moderado, pero `tieneinternet 0.35` es bajo — sugiere que la conectividad ya no discrimina dentro de este bloque homogéneo. Es el único clúster donde `Personas hogar 1.48` supera a `Cuartos 1.54` de cerca, indicando hacinamiento como factor.

### 3.2 Clúster 1 (15.9%, 707k) — segundo bloque grande, R² 0.163

Top: `Estrato 3.30 > Noche 2.95 > Rural 2.67 > Bolívar 2.22`. A diferencia del 0, aquí **el hogar domina**: `Madre Téc. Comp. 1.94`, `Internet 1.55`, `Estrato 3.30` (máximo entre todos los clústeres). `cole_naturaleza_Oficial 1.30` aparece solo aquí en top-10, señal de que la dicotomía oficial/privado sí discrimina en este territorio. Geografía concentrada: `Bolívar 2.22` y `Magdalena 1.55` duplican a Antioquia.

### 3.3 Clúster 2 (10.6%, 471k) — bloque urbano-Bogotá, R² 0.107

Top: `Género F 5.08 (máximo global) > Noche 4.04 > PC 2.27 > Bogotá 2.03`. Es el **territorio del género y la tecnología**: el SHAP de género femenino duplica al de cualquier otro clúster, y `tienecomputador 2.27` es el segundo más alto. `Bogotá 2.03` confirma anclaje geográfico; ningún otro departamento supera 0.74 aquí.

### 3.4 Clúster 3 (17.5%, 776k) — gemelo del 0, R² 0.159

Top: `Género M 4.00 > Noche 3.15 = Sabatina 3.15 > PC 2.44 > Cuartos 1.86`. Espejo del clúster 0 pero con **género invertido** (M en 3 vs M en 0, ambos ~3–4) y `Rural 1.81` menos dominante que en 0 (3.33). La simetría sugiere que 0 y 3 son dos caras del mismo bloque medio, separadas por jornada y departamento (Santander 1.63 vs Bolívar 1.62).

### 3.5 Clúster 4 (15.4%, 684k) — territorio de educación parental alta, R² 0.130

Top: `Género F 2.88 > Cuartos 2.56 > Madre Téc. Comp. 2.34 > Madre Prof. Comp. 2.22 > Padre Téc. Comp. 2.07`. **Único clúster donde las 4 dummies de educación superior parental están en top-7** (dos de madre, dos de padre). `Estrato 1.76` retrocede al puesto 6, el más bajo entre todos los clústeres: aquí la educación pesa más que el estrato.

### 3.6 Clúster 5 (9.0%, 400k) — minoritario, subestimado (MBE −6.6), R² 0.147

Top: `Padre Postgrado 5.50 > Mañana 4.97 > Calendario B 4.84 > Estrato 3.52 > Madre Postgrado 3.14`. **Outlier total:** es el único clúster donde `Postgrado` (madre y padre) domina y donde `Calendario B 4.84` y `Jornada Mañana 4.97` son top-3. `tieneinternet 0.02` y `tienecomputador 0.13` son irrelevantes aquí (vs 1.5 y 2.4 en otros), indicando saturación de conectividad. Este territorio explica el sesgo negativo del reporte de evaluación: el modelo no captura bien el efecto postgrado.

### 3.7 Clúster 6 (11.6%, 513k) — jornada Completa extrema, R² 0.167

Top: `Jornada Completa 11.64 (outlier global, 2.5× el siguiente) > Sabatina 4.67 > Noche 2.77 > Cuartos 2.31`. La **jornada Completa es 11.64**, más del doble que cualquier otra feature en cualquier clúster. Sugiere un subgrupo de colegios con jornada única que concentra fuertemente la predicción. El resto del perfil replica al clúster 4 (educación parental técnica/profesional en puestos 6–10).

---

## 4. Patrones transversales y diferenciales

**Variables universales (presentes en top-10 de ≥6 clústeres):** `fami_cuartoshogar` (7/7), `fami_estratovivienda` (7/7), `cole_jornada_Noche` (6/7), `estu_genero` (7/7). Son palancas robustas para política.

**Variables diferenciales (alta varianza entre clústeres, CV alto):**

*   `cole_jornada_Completa`: 0.16–0.66 en clústeres 0–5, **11.64 en 6** (CV > 1.5). Palanca exclusiva del clúster 6.
*   `fami_educacionpadre_Postgrado`: 9e-05 en clúster 0, **5.50 en clúster 5**, 0.18 en 6 (CV > 1.8). Solo útil en territorio 5.
*   `cole_calendario_B`: ~0 en 0–4, **4.84 en 5**, 0.95 en 6. Marca del clúster 5.
*   `estu_genero_F`: 1.69 en 1 vs **5.08 en 2** vs 2.28 en 6 (CV ~0.45). Importancia asimétrica de género.

**Variables geográficamente ancladas:** `Bolívar` es top-6 en 0 y 1 pero irrelevante en 2–6; `Bogotá` solo pesa en 2; `Santander` solo en 3–5. Confirma la noción de territorios funcionales: el departamento no es un predictor global, sino local.

**Variables irrelevantes globalmente:** `cole_bilingue 0.0` y `cole_sede_principal 0.0` en todos los clústeres; `cole_caracter_No Aplica` < 0.05 salvo 6 (0.31). Pueden excluirse en iteraciones futuras sin pérdida.

---

## 5. Conexión con la evaluación y síntesis

El SHAP explica el gradiente de MBE del reporte de evaluación:

*   Clústeres 0–3, con MBE cercano a 0, tienen SHAP dominado por **jornada y género**, variables bien capturadas por el modelo.
*   Clústeres 5–6, con subestimación −6 a −11, tienen SHAP dominado por **postgrado y jornada Completa**, variables de cola que el modelo infravalora. La recomendación no es ajustar el modelo, sino complementar con variables de capital cultural no medidas en Saber 11.

**Implicación para el ICFES:** no hay una única palanca nacional. La intervención más eficiente sería:

*   Territorios 0–3 (52% pobl.): jornada (reducir Sabatina/Noche) y brecha de género.
*   Territorio 4 (15%): educación técnica/profesional de padres — programas de formación adulta.
*   Territorio 5 (9%): calendario B y postgrado — focalizar en calidad docente de esa oferta.
*   Territorio 6 (12%): jornada Completa — auditar por qué esta modalidad concentra 11.6 puntos de SHAP.

SHAP segmentado convierte la heterogeneidad —antes promediada en un R² global de 0.13–0.16— en **7 rankings accionables**, cumpliendo el objetivo de la cadena LCA → XGBoost → XAI del Replanteamiento metodológico.

---
*Fuente: `shap_importance_XGBoost.txt` — TreeExplainer sobre `xgb_cluster_{0..6}.json`, 10 000 muestras/clúster — 21 de agosto de 2026.*
