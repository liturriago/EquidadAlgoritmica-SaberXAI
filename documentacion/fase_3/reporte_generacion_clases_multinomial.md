# Reporte: Generación de Archivos clase_N.parquet (Multinomial)

**Fecha:** 21 de agosto de 2026  
**Script utilizado:** `bases_datos/preprocesar_clases_lca.py`

## Contexto

Se actualizó el archivo fuente de datos de `saber11_lca_clases.parquet` a `saber11_lca_clases_multinomial.parquet`, el cual contiene clasificaciones LCA multinomiales con 7 clases (0-6) en lugar de las clases anteriores.

## Cambios Realizados

### 1. Actualización del Script

Se modificó `bases_datos/preprocesar_clases_lca.py` para trabajar con el nuevo archivo:

- **Archivo de entrada:** `saber11_lca_clases_multinomial.parquet`
- **Columnas de clase:** `clase_lca_mult` y `prob_max_lca_mult` (antes `clase_lca` y `prob_max_lca`)
- **Columnas eliminadas:** Se mantienen las mismas columnas de clase/probabilidad en el drop

### 2. Estructura del Nuevo Dataset

**Archivo fuente:** `data/saber11_lca_clases_multinomial.parquet`
- **Total de registros:** 4,445,422 filas
- **Total de columnas:** 23
- **Clases LCA:** [0, 1, 2, 3, 4, 5, 6] (7 clases)

**Columnas incluidas:**
- Features sociodemográficas del colegio (9 columnas)
- Features familiares y del hogar (10 columnas)
- Puntaje global (`punt_global`)
- Clase LCA multinomial (`clase_lca_mult`)
- Probabilidad máxima (`prob_max_lca_mult`)

### 3. Transformaciones Aplicadas

El script aplica las siguientes transformaciones:

**Variables binarias** (Si/No → 1/0):
- `fami_tieneautomovil`, `fami_tienecomputador`, `fami_tieneinternet`
- `fami_tienelavadora`, `fami_tieneserviciotv`
- `cole_bilingue`, `cole_sede_principal`

**Variables ordinales** (texto → numérico):
- `fami_estratovivienda`: Sin Estrato=0, Estrato 1-6=1-6
- `fami_cuartoshogar`: Uno=1, Dos=2, ..., Seis O Más=6
- `fami_personashogar`: 1 A 2=1, 3 A 4=3, ..., 9 O Más=9

**Variables categóricas** (texto → Categorical):
- `cole_area_ubicacion`, `cole_calendario`, `cole_caracter`
- `cole_depto_ubicacion`, `cole_genero`, `cole_jornada`
- `cole_naturaleza`, `estu_genero`

## Resultados

Se generaron 7 archivos parquet en `data/`, uno por cada clase LCA:

| Archivo | Filas | Tamaño (MB) | Porcentaje |
|---------|-------|-------------|------------|
| clase_0.parquet | 891,146 | 3.82 | 20.0% |
| clase_1.parquet | 707,887 | 3.10 | 15.9% |
| clase_2.parquet | 471,189 | 1.85 | 10.6% |
| clase_3.parquet | 776,372 | 3.34 | 17.5% |
| clase_4.parquet | 684,733 | 2.90 | 15.4% |
| clase_5.parquet | 400,113 | 1.53 | 9.0% |
| clase_6.parquet | 513,982 | 2.20 | 11.6% |
| **Total** | **4,445,422** | **18.74** | **100%** |

**Compresión:** zstd (alta compresión, lectura rápida)

## Observaciones

- La distribución de clases es relativamente balanceada, con la clase 0 siendo la más frecuente (20%) y la clase 5 la menos frecuente (9%)
- Cada archivo parquet incluye todas las features transformadas más `punt_global`
- Los archivos están listos para entrenamiento de modelos XGBoost/MLP por clase
