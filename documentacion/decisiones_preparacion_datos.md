# Decisiones de preparación de datos – no cubiertas en feature_selection.md

Este documento recoge decisiones técnicas aplicadas durante la construcción del dataset limpio que no quedaron documentadas en `feature_selection.md` ni en `exclusion_2014_1_columnas_comunes.md`.

---

## 1. Normalización de texto geográfico (`strip_accents`)

### Decisión
Se aplica una función de normalización Unicode (NFKD → ASCII) seguida de `.strip().title()` a las columnas geográficas:
- `cole_depto_ubicacion`
- `cole_mcpio_ubicacion`
- `estu_depto_reside`

### Impacto medido

| Columna | Antes | Después | Reducción |
|---|---|---|---|
| `cole_depto_ubicacion` | 34 valores únicos | 33 | −1 |
| `cole_mcpio_ubicacion` | 1 396 valores únicos | 1 081 | −315 |
| `estu_depto_reside` | 35 valores únicos | 34 | −1 |

### Razón
El ICFES registra nombres con y sin tilde según el periodo (e.g., "Bogotá" vs "Bogota", "Córdoba" vs "Cordoba"). Sin esta normalización, el modelo aprende representaciones distintas para el mismo lugar, contaminando los embeddings geográficos y los análisis de brecha departamental.

---

## 2. Descomposición de `periodo` en `anio` + `semestre`

### Decisión
El campo `periodo` (entero de 5 dígitos, e.g., `20142`) se descompone en dos variables:
- `anio` (int): los primeros 4 caracteres → `2014`
- `semestre` (str): el último carácter → `"2"`

`periodo` se elimina del dataset final (documentado en feature_selection.md como "redundante").

### Razón
La descomposición permite al modelo capturar:
1. **Tendencia temporal continua** a través de `anio` (efectos de política educativa, pandemia COVID-19 en 2020-2021).
2. **Efecto de semestre** de forma independiente: el semestre 1 (primer semestre del año) y el semestre 2 pueden tener perfiles de estudiantes distintos.

Mantener `periodo` como entero (20142, 20151…) crearía una variable con saltos irregulares (20181 → 20182 → 20191) que el modelo trataría incorrectamente como escala numérica continua.

---

## 3. Variables conservadas no listadas en `feature_selection.md`

El dataset final tiene 28 columnas. `feature_selection.md` lista 27 (incluyendo el target). Las dos variables siguientes están en el dataset pero ausentes de la tabla de conservadas:

### `cole_mcpio_ubicacion`
- **Qué es:** nombre del municipio donde está ubicado el colegio.
- **Por qué se conservó:** aporta granularidad geográfica dentro de cada departamento, relevante para análisis de brechas entre municipios capitales y no capitales. `cole_depto_ubicacion` sola no captura esta variación intra-departamental.
- **Riesgo:** alta cardinalidad (1 081 valores únicos tras normalización). Se asume que los embeddings del modelo de deep learning manejarán esto; de lo contrario puede ser necesario agrupar por tamaño de municipio o eliminarla en etapas de modelado.

### `estu_tipodocumento`
- **Qué es:** tipo de documento de identidad del estudiante (CC, TI, CE, pasaporte, etc.).
- **Por qué se conservó:** actúa como proxy del estatus migratorio o edad legal del estudiante. Estudiantes con Cédula de Extranjería (CE) o pasaporte representan una población con condiciones de acceso y contexto socioeconómico diferenciado, relevante para el análisis de equidad.
- **Cardinalidad:** baja (menos de 10 categorías). No genera presión en los embeddings.

---

## 4. Criterio de split urbano/rural: `cole_area_ubicacion` vs `estu_depto_reside`

### Decisión
Los subconjuntos `v2_datos_urbano.csv` y `v2_datos_rural.csv` se generan filtrando por `cole_area_ubicacion` (área de ubicación del **colegio**), no por el lugar de residencia del estudiante.

### Distribución resultante

| Área | Filas | Porcentaje |
|---|---|---|
| Urbano | 4 185 434 | 86.82% |
| Rural | 635 603 | 13.18% |
| **Total** | **4 821 037** | **100%** |

### Razón
El área del colegio define el contexto educativo que el modelo intentará explicar: infraestructura disponible, cuerpo docente, jornada escolar. La residencia del estudiante es relevante para acceso, pero muchos estudiantes rurales asisten a colegios urbanos o viceversa. Usar el área del colegio mantiene la coherencia entre el target (puntaje obtenido en ese colegio) y el contexto institucional.

---

## 5. Lectura de archivos crudos con `dtype=str`

### Decisión
En `merge_columnas_comunes.py`, todos los archivos `.txt` se leen con `dtype=str`:
```python
df = pd.read_csv(archivo, sep=";", dtype=str, encoding="utf-8")
```

### Razón
Los 21 periodos tienen esquemas distintos y pandas inferiría tipos de forma inconsistente entre archivos. Por ejemplo, un campo que en 2014-2 tiene solo valores numéricos sería leído como `int64`, pero en 2020-1 podría incluir "N/A" y se leería como `object`. Forzar `dtype=str` garantiza que la concatenación no falle ni produzca coerciones silenciosas. La conversión a tipos definitivos ocurre después del merge en el notebook de presentación.

---

## 6. Filtro de edad 14–60 al derivar `edad_examen`

### Decisión
En el notebook exploratorio (`02_estudio_datos.ipynb`), la edad al momento del examen se calcula como:
```python
edad_examen = anio - estu_fechanacimiento.year
```
Solo se conservan registros con `edad_examen` entre 14 y 60.

### Razón
El campo `estu_fechanacimiento` contiene errores de captura frecuentes (años imposibles como 1900, 2090, o fechas con formato inválido). El rango 14–60 cubre:
- Mínimo realista para grado 11 (14 años).
- Máximo razonable para validantes de bachillerato adultos (grado 26), que son una población legítima en el dataset.

Este filtro es solo analítico (notebook exploratorio) y no se aplica al dataset de entrenamiento.

---

## 7. Normalización de etiquetas categóricas en carga

### Decisión
Al cargar el dataset en los notebooks, se aplica normalización de cadenas a varias columnas antes de cualquier análisis:

```python
df["cole_area_ubicacion"] = df["cole_area_ubicacion"].str.strip().str.title()
df["cole_naturaleza"]     = df["cole_naturaleza"].str.strip().str.title()
df["estu_genero"]         = df["estu_genero"].str.strip().str.upper()
```

### Razón
Los archivos crudos del ICFES presentan inconsistencias de capitalización entre periodos (e.g., "URBANO", "Urbano", "urbano"). La normalización es necesaria antes de cualquier agrupación o filtro, y debe aplicarse de forma consistente para evitar que `value_counts()` o `groupby()` devuelvan categorías duplicadas.

---

## 8. Codificación ordinal manual para correlación de Spearman

### Decisión
En el análisis exploratorio, las variables categóricas ordinales se codifican con mapeos manuales antes de calcular correlación de Spearman con `punt_global`:

| Variable | Escala |
|---|---|
| `fami_estratovivienda` | Sin Estrato=0, Estrato 1=1, …, Estrato 6=6 |
| `cole_area_ubicacion` | Rural=0, Urbano=1 |
| `cole_naturaleza` | Oficial=0, No Oficial=1 |
| `fami_tieneX` | No=0, Si=1 |
| `fami_educacionmadre/padre` | Ninguno=0, …, Postgrado=9 |
| `fami_personashogar` | Una=1, Dos=2, …, Diez O Más=10 |

### Razón
Spearman requiere valores numéricos. Los mapeos respetan el orden natural de las categorías (en especial educación y estrato), lo que hace la correlación interpretable como "a mayor nivel, mayor/menor puntaje". Usar one-hot encoding para este análisis produciría correlaciones sin sentido direccional.

Estos mapeos son solo para el análisis exploratorio. El modelo de deep learning usará embeddings o codificación separada.
