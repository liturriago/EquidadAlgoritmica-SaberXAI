# Selección de Variables – ICFES Saber 11

## Contexto del proyecto

Este proyecto aplica deep learning sobre el dataset ICFES Saber 11 para predecir `punt_global` como variable target. El objetivo central es dar **interpretabilidad al modelo** para analizar brechas de equidad: por área geográfica (urbano/rural), estrato socioeconómico, género, tipo de colegio y departamento.

La selección de variables busca tres cosas:
1. Eliminar data leakage que haría el modelo trivial.
2. Conservar únicamente features que expliquen condiciones del estudiante, no sus resultados.
3. Garantizar que los métodos de interpretabilidad (SHAP u otros) reflejen factores reales de equidad.

---

## Variables eliminadas

### 1. Data leakage — sub-puntajes por área

| Variable | Razón |
|---|---|
| `punt_matematicas` | Componente directo de `punt_global` |
| `punt_lectura_critica` | Componente directo de `punt_global` |
| `punt_c_naturales` | Componente directo de `punt_global` |
| `punt_sociales_ciudadanas` | Componente directo de `punt_global` |
| `punt_ingles` | Componente directo de `punt_global` |
| `desemp_ingles` | Nivel de desempeño derivado de `punt_ingles` |

`punt_global` en ICFES se calcula como función directa de estos sub-puntajes. Incluirlos como features no genera aprendizaje útil: el modelo simplemente reconstruiría la fórmula de ICFES y los métodos de interpretabilidad mostrarían que "matemáticas importa" por construcción matemática, no por hallazgos de equidad.

---

### 2. Identificadores administrativos

| Variable | Razón |
|---|---|
| `estu_estudiante` | Número de documento del estudiante — identificador puro |
| `estu_consecutivo` | Número secuencial ICFES por periodo — identificador puro |
| `cole_cod_dane_establecimiento` | Código DANE del colegio — sin valor predictivo |
| `cole_cod_dane_sede` | Código DANE de la sede — sin valor predictivo |
| `cole_cod_depto_ubicacion` | Código numérico del departamento — redundante con `cole_depto_ubicacion` |
| `cole_cod_mcpio_ubicacion` | Código numérico del municipio — redundante con `cole_mcpio_ubicacion` |
| `cole_codigo_icfes` | Código interno ICFES del colegio — sin valor predictivo |
| `estu_cod_depto_presentacion` | Código numérico — redundante con `estu_depto_presentacion` |
| `estu_cod_mcpio_presentacion` | Código numérico — redundante con `estu_mcpio_presentacion` |
| `estu_cod_reside_depto` | Código numérico — redundante con `estu_depto_reside` |
| `estu_cod_reside_mcpio` | Código numérico — redundante con `estu_mcpio_reside` |

---

### 3. Nombres de alta cardinalidad

| Variable | Razón |
|---|---|
| `cole_nombre_establecimiento` | Texto libre con miles de valores únicos — no codificable con sentido para DL |
| `cole_nombre_sede` | Ídem — la información geográfica relevante ya está en `cole_depto_ubicacion` |

---

### 4. Redundantes con otras columnas conservadas

| Variable | Redundante con | Razón |
|---|---|---|
| `periodo` | `anio` + `semestre` | Ya se descompuso en variables más útiles |
| `estu_depto_presentacion` | `cole_depto_ubicacion` | El lugar de presentación del examen coincide con la ubicación del colegio en la gran mayoría de casos |
| `estu_mcpio_presentacion` | `cole_depto_ubicacion` | Ídem, mayor cardinalidad sin ganancia informativa |
| `estu_mcpio_reside` | `estu_depto_reside` | El departamento de residencia es suficiente para capturar geografía; municipio añade cardinalidad sin beneficio proporcional |

---

### 5. Varianza casi nula

| Variable | Distribución | Razón |
|---|---|---|
| `estu_privado_libertad` | 100% N, ~0% S | Sin varianza útil para el modelo |
| `estu_pais_reside` | 99.57% COLOMBIA | 97 valores únicos pero concentrado en uno — no aporta señal |
| `estu_nacionalidad` | 99.43% COLOMBIA | Ídem |

---

### 6. Índices socioeconómicos compuestos

| Variable | Razón |
|---|---|
| `estu_inse_individual` | INSE (Índice de Nivel Socioeconómico) calculado por ICFES a partir de las variables `fami_*`. Redundante si conservamos las variables base. |
| `estu_nse_individual` | NSE individual — mismo argumento que INSE. |

Se prefieren las variables `fami_*` desagregadas porque son más interpretables para el análisis de equidad: permiten identificar qué dimensión específica del nivel socioeconómico (educación de padres, bienes del hogar, estrato) tiene mayor impacto.

---

### 7. Flags administrativos derivados

| Variable | Razón |
|---|---|
| `estu_agregado` | Indica si el estudiante entra en estadísticas agregadas del colegio. Es un flag derivado de condiciones ya capturadas por otras variables (grado, situación especial). No es una causa del desempeño. |

---

## Variables conservadas (27)

| Grupo | Variables |
|---|---|
| **Colegio** | `cole_area_ubicacion`, `cole_bilingue`, `cole_calendario`, `cole_caracter`, `cole_depto_ubicacion`, `cole_genero`, `cole_jornada`, `cole_naturaleza`, `cole_sede_principal` |
| **Estudiante** | `estu_depto_reside`, `estu_fechanacimiento`, `estu_genero`, `estu_grado` |
| **Familia** | `fami_estratovivienda`, `fami_educacionmadre`, `fami_educacionpadre`, `fami_cuartoshogar`, `fami_personashogar`, `fami_tienecomputador`, `fami_tieneinternet`, `fami_tienelavadora`, `fami_tieneautomovil`, `fami_tieneserviciotv` |
| **Temporal** | `anio`, `semestre` |
| **Target** | `punt_global` |

### Nota sobre `estu_grado`

Se conserva a pesar de que el 88% de los registros corresponden a grado 11. El 11.91% restante corresponde al **grado 26 (validantes de bachillerato)**: adultos que certifican su bachillerato a través del examen sin haber cursado grado 11 en una institución. Esta es una población con características socioeconómicas y de desempeño sistemáticamente distintas, relevante para el análisis de equidad.

---

## Impacto en memoria y relevancia para deep learning

### Reducción de peso en RAM

| | Columnas | Peso en memoria |
|---|---|---|
| Dataset original | 57 | 12,706.9 MB (~12.4 GB) |
| Dataset limpio | 27 | 7,676.3 MB (~7.5 GB) |
| **Reducción** | **−29 columnas** | **−5,030.6 MB (~39.6%)** |

### Por qué esto importa en deep learning

**Tiempo de entrenamiento.** Cada epoch procesa todos los registros del dataset. Cargar ~5 GB menos por epoch reduce el tiempo de I/O y el movimiento de datos entre RAM y GPU/CPU de forma acumulativa a lo largo de cientos de epochs.

**Presión sobre la VRAM.** En deep learning, los batches se mueven a memoria de GPU. Un dataset más liviano permite usar batch sizes más grandes, lo que estabiliza el gradiente y acelera la convergencia. Con 12 GB en RAM, mantener el dataset completo en VRAM de una GPU estándar (8–16 GB) era inviable; con 7.5 GB hay más margen.

**Ruido en los gradientes.** Las columnas eliminadas (identificadores, flags, variables de baja varianza) no aportan señal pero sí ruido durante el backpropagation. Cada columna irrelevante obliga a la red a aprender que su peso debe ser ~0, desperdiciando capacidad del modelo y ralentizando la convergencia.

**Embeddings categoriales.** Las redes neuronales típicamente representan variables categóricas como embeddings. Eliminar columnas de alta cardinalidad sin semántica útil (`cole_nombre_establecimiento` con miles de valores únicos) evita embeddings masivos que consumirían parámetros sin aportar poder predictivo.

**Reproducibilidad y velocidad en Kaggle.** Al subir el dataset limpio a Kaggle, los notebooks de entrenamiento parten directamente del dataset procesado, evitando repetir la limpieza en cada ejecución y reduciendo el tiempo de setup de cada experimento.

---

## Resumen

| Categoría | Columnas eliminadas |
|---|---|
| Data leakage | 6 |
| Identificadores administrativos | 11 |
| Alta cardinalidad | 2 |
| Redundantes | 4 |
| Baja varianza | 3 |
| Compuestos redundantes | 2 |
| Flags administrativos | 1 |
| **Total eliminadas** | **29** |
| **Total conservadas** | **27** |
