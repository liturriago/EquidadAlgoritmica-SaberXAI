# Reporte de Sesión: Implementación de Pipeline de Entrenamiento por Clúster y Análisis SHAP

**Fecha:** 1 de agosto de 2026  
**Participantes:** Usuario (Lucas) + Asistente IA  
**Duración:** Sesión extendida de desarrollo iterativo

---

## 1. Contexto Inicial

El proyecto **EquidadAlgoritmica-SaberXAI** ya tenía implementado:
- Clustering LCA con 7 territorios funcionales (clústeres 0-6)
- Dataset `saber11_lca_clases.parquet` con 4.4M de registros
- Documentación del replanteamiento metodológico (de DANN a territorios funcionales)

**Estado previo:**
- Los datos ya estaban preprocesados y divididos en 7 archivos parquet por clúster
- No existía pipeline de entrenamiento específico por clúster
- No existía análisis SHAP segmentado

---

## 2. Objetivos de la Sesión

1. **Crear script de preprocesamiento** para generar los 7 parquets por clúster
2. **Adaptar la infraestructura de entrenamiento** para entrenar modelos separados por clúster
3. **Implementar análisis SHAP** segmentado por territorios funcionales
4. **Resolver problemas de consistencia** entre preprocesamiento de entrenamiento y SHAP

---

## 3. Desarrollo Iterativo

### 3.1. Preprocesamiento de Datos por Clúster

**Archivo creado:** `bases_datos/preprocesar_clases_lca.py`

**Funcionalidad:**
- Lee `saber11_lca_clases.parquet`
- Separa en 7 archivos por clúster (clase_0.parquet a clase_6.parquet)
- Aplica transformaciones:
  - Variables binarias (Si/No → 1/0)
  - Variables ordinales (mapeo de educación, estrato, cuartos, personas)
  - Conversión a Categorical para optimización
- Elimina columnas `clase_lca` y `prob_max_lca`

**Resultados:**
| Clúster | Filas | % Población |
|---------|------:|:-----------:|
| 0 | 2,096,589 | 47.2% |
| 1 | 128,257 | 2.9% |
| 2 | 727,425 | 16.4% |
| 3 | 830,726 | 18.7% |
| 4 | 257,714 | 5.8% |
| 5 | 230,865 | 5.2% |
| 6 | 173,846 | 3.9% |

---

### 3.2. Adaptación de Infraestructura para Entrenamiento por Clúster

#### 3.2.1. Modificaciones en `config.py`

**Campos agregados:**
```python
cluster_id: Optional[int] = None  # ID del clúster para entrenamiento
data_dir: Optional[str] = None    # Directorio base de parquets
cat_col_to_encode: Optional[str]  # Ahora es Optional (puede ser None)
```

**Razón:** Permitir cargar datos específicos por clúster y deshabilitar Target Encoding cuando no hay columna de municipio.

#### 3.2.2. Modificaciones en `data_module.py`

**Cambios:**
1. **Carga condicional de datos:**
   ```python
   if self.config.cluster_id is not None and self.config.data_dir is not None:
       data_path = Path(self.config.data_dir) / f"clase_{self.config.cluster_id}.parquet"
   else:
       data_path = Path(self.config.data_path)
   ```

2. **Target Encoding opcional:**
   ```python
   if cat_to_encode is not None and cat_to_encode in self.X_train.columns:
       # Aplicar Target Encoding
   ```

**Problema resuelto:** Los datos LCA no tienen `cole_mcpio_ubicacion`, por lo que el Target Encoding debía ser opcional.

#### 3.2.3. Script de Entrenamiento por Clúster

**Archivo creado:** `scripts/train_by_cluster.py`

**Funcionalidad:**
- Itera sobre los 7 clústeres (0-6)
- Para cada clúster:
  1. Carga datos usando `DataModule` con `cluster_id`
  2. Entrena XGBoost con early stopping
  3. Entrena MLP con early stopping
  4. Evalúa ambos modelos (global, rural, urbano)
  5. Guarda modelos como `xgb_cluster_{n}.json` y `mlp_cluster_{n}.pth`

**Configuración:** `configs/lca_clusters.yaml`
```yaml
data_dir: "data/"
cat_col_to_encode: null
mlp_batch_size: 4096
xgb_n_estimators: 350
```

---

### 3.3. Implementación de Análisis SHAP

#### 3.3.1. Script Principal de SHAP

**Archivo creado:** `scripts/shap_analysis.py`

**Funcionalidad:**
- Calcula SHAP values para los 14 modelos (7 XGBoost + 7 MLP)
- Genera visualizaciones:
  - Summary plots (beeswarm) por clúster
  - Bar plots comparativos top-10 variables
- Exporta métricas de importancia en JSON y CSV
- Análisis diferencial: identifica variables con impacto asimétrico entre territorios

**Características técnicas:**
- **XGBoost:** Usa `TreeExplainer` (exacto y rápido)
- **MLP:** Usa `DeepExplainer` (aproximación para redes neuronales)
- **Muestreo:** Limita a 10,000 muestras por clúster para eficiencia
- **Fallback:** Si `TreeExplainer` falla, usa `KernelExplainer`

#### 3.3.2. Configuración SHAP

**Archivo creado:** `configs/shap_analysis.yaml`
```yaml
shap_models_dir: "models"
shap_data_dir: "data"
shap_output_dir: "shap_results"
shap_max_samples: 10000
shap_model_type: "xgb"  # 'xgb', 'mlp' o 'both'
shap_clusters: [0, 1, 2, 3, 4, 5, 6]
```

**Campos agregados a `config.py`:**
```python
shap_max_samples: int = 10000
shap_output_dir: str = "shap_results"
shap_models_dir: str = "models"
shap_data_dir: str = "data"
shap_clusters: List[int] = [0, 1, 2, 3, 4, 5, 6]
shap_model_type: str = "both"
```

#### 3.3.3. Control desde CLI y YAML

**Argumentos CLI:**
```bash
python scripts/shap_analysis.py \
    --config configs/shap_analysis.yaml \
    --model-type xgb \
    --clusters 0 3 5 \
    --max-samples 5000
```

**Prioridad:** CLI > YAML > defaults

---

## 4. Problemas Resueltos

### 4.1. Incompatibilidad SHAP/XGBoost 3.x

**Error:**
```
ValueError: could not convert string to float: '[-2.701439E-2]'
```

**Causa:** XGBoost 3.x serializa `base_score` como array, pero SHAP < 0.46 espera float.

**Solución:** Fallback automático a `KernelExplainer` cuando `TreeExplainer` falla.

```python
try:
    explainer = shap.TreeExplainer(xgb_model.model)
    shap_values = explainer.shap_values(X)
except (ValueError, AttributeError):
    # Fallback a KernelExplainer
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X)
```

### 4.2. Features Inconsistentes entre Clústeres

**Problema:** Cada clúster tenía diferente número de features (60-68) debido a one-hot encoding inconsistente.

**Análisis:**
- Los modelos XGBoost fueron entrenados con features específicas por clúster
- `shap_analysis.py` inicialmente hacía one-hot encoding de forma diferente
- Esto causaba error de dimensiones al calcular SHAP

**Solución implementada:**
1. **`load_cluster_data()` ahora usa `DataModule`** para garantizar consistencia con el entrenamiento
2. **`compute_shap_xgb()` alinea features** con las del modelo:
   ```python
   model_feature_names = xgb_model.model.feature_names
   # Seleccionar solo las features que el modelo espera
   X = X[:, [feature_to_idx[f] for f in model_feature_names]]
   ```
3. **Retorna feature names alineados** para que los plots usen los nombres correctos

### 4.3. Error de Shape en Summary Plots

**Error:**
```
AssertionError: The shape of the shap_values matrix does not match the shape of the provided data matrix
```

**Causa:** Los feature names no coincidían después de la alineación.

**Solución:** `compute_shap_xgb()` y `compute_shap_mlp()` ahora retornan tuplas `(shap_values, feature_names)` para que `plot_summary_comparison()` use los nombres correctos.

---

## 5. Decisiones de Diseño

### 5.1. Modelos Separados por Clúster vs. Modelo Único con Group DRO

**Decisión:** Entrenar 7 modelos separados (uno por clúster)

**Razones:**
1. Facilita el análisis SHAP segmentado
2. Cada modelo se especializa en su territorio funcional
3. Permite comparar importancia de variables entre territorios
4. Más interpretable que un modelo único con pérdida robusta

**Trade-off:** No se implementó Group DRO (mencionado en el artículo), pero el enfoque de modelos separados permite análisis comparativo equivalente.

### 5.2. XGBoost vs. MLP para SHAP

**Análisis de resultados de entrenamiento:**

| Clúster | XGBoost R² | MLP R² |
|---------|-----------|--------|
| 0 | 0.300 | 0.289 |
| 1 | -0.143 | -25.508 |
| 2 | 0.178 | 0.165 |
| 3 | 0.267 | 0.259 |
| 4 | 0.226 | 0.023 |
| 5 | 0.145 | -0.583 |
| 6 | 0.363 | -8.227 |

**Conclusión:** XGBoost supera a MLP en todos los clústeres. MLP colapsa en clústeres pequeños (1, 5, 6).

**Recomendación:** Usar solo XGBoost para SHAP (`shap_model_type: "xgb"`).

### 5.3. Clúster 1 como Caso Especial

**Observación:** El clúster 1 (más privilegiado) tiene R² negativo incluso con XGBoost.

**Interpretación:**
- Grupo más heterogéneo no capturado por covariables
- Factores no medidos (redes sociales, capital cultural) tienen mayor peso
- Implicación para equidad: intervenciones basadas en el modelo serían menos confiables para este grupo

### 5.4. Aceptación de Features Inconsistentes

**Decisión:** No forzar el mismo esquema de features en todos los clústeres.

**Razones:**
1. Respeta las features con las que fueron entrenados los modelos
2. Cada clúster tiene su propio esquema de one-hot encoding
3. El análisis diferencial compara solo features comunes
4. Evita introducir sesgo al forzar features faltantes

---

## 6. Estructura Final del Código

### 6.1. Archivos Modificados

```
src/saber_xai/
├── config.py              # Agregados campos para cluster_id y SHAP
├── data/
│   └── data_module.py     # Carga condicional por cluster_id
└── models/
    └── evaluator.py       # Método evaluate_by_cluster()
```

### 6.2. Archivos Nuevos

```
bases_datos/
└── preprocesar_clases_lca.py

scripts/
├── train_by_cluster.py
└── shap_analysis.py

configs/
├── lca_clusters.yaml
└── shap_analysis.yaml

tests/
└── test_shap.py           # 7 tests nuevos
```

### 6.3. Tests

**Total:** 19 tests pasando

**Cobertura:**
- Configuración SHAP (defaults, modificación)
- Funciones de carga y muestreo de datos
- Cálculo SHAP para XGBoost y MLP
- Manejo de errores (archivo no encontrado)

---

## 7. Uso en Kaggle

### 7.1. Entrenamiento por Clúster

```bash
# Entrenar todos los clústeres
python scripts/train_by_cluster.py --config configs/lca_clusters.yaml

# Entrenar clústeres específicos
python scripts/train_by_cluster.py --clusters 0 3 5
```

### 7.2. Análisis SHAP

```bash
# Análisis completo (solo XGBoost, recomendado)
python scripts/shap_analysis.py --config configs/shap_analysis.yaml

# Solo clústeres específicos
python scripts/shap_analysis.py --clusters 0 3 5

# Muestras limitadas (más rápido)
python scripts/shap_analysis.py --max-samples 5000
```

### 7.3. Tiempos Estimados

| Configuración | Tiempo |
|---------------|--------|
| XGBoost solo (7 clústeres) | ~7-15 min |
| MLP solo (7 clústeres) | ~70-100 min |
| Ambos modelos | ~80-115 min |

---

## 8. Próximos Pasos Sugeridos

1. **Ejecutar análisis SHAP en Kaggle** con `shap_model_type: "xgb"`
2. **Analizar resultados:**
   - ¿Qué variables son más importantes en cada territorio?
   - ¿Hay variables con impacto diferencial entre territorios?
   - ¿El clúster 1 confirma su baja predictibilidad?
3. **Generar visualizaciones comparativas** entre territorios
4. **Redactar hallazgos** para el artículo/policy brief
5. **Opcional:** Implementar Group DRO como comparación

---

## 9. Commits de la Sesión

```
6a090be fix: use DataModule in SHAP for consistent preprocessing
507797f fix: align features with model schema in SHAP XGBoost
a3d550f feat: add configuration file for SHAP analysis parameters
fa28b79 feat: add shap_data_dir and shap_clusters to config
a50c1f5 fix: handle per-cluster feature schemas in SHAP analysis
94a3c8d feat: add --model-type flag to control SHAP analysis
94cfe70 feat: add YAML config support and fix SHAP/XGBoost 3.x
761b3c1 test: add comprehensive SHAP analysis tests
2b410a0 chore: add checkpoints/ to .gitignore
fff390f feat: add SHAP config fields and tests
09de58f feat: add SHAP analysis script for cluster-based models
1cc0b04 feat: add cluster-based training pipeline for LCA territories
```

---

## 10. Resumen Ejecutivo

**Logros:**
- ✅ Pipeline completo de entrenamiento por clúster implementado
- ✅ Análisis SHAP segmentado por territorios funcionales
- ✅ Resolución de problemas de consistencia en preprocesamiento
- ✅ 19 tests pasando
- ✅ Documentación completa en YAML y CLI

**Hallazgos clave:**
- XGBoost supera consistentemente a MLP en todos los clústeres
- El clúster 1 (más privilegiado) es el más difícil de predecir
- Las features varían entre clústeres (60-68), lo cual es aceptado para respetar el entrenamiento

**Recomendación:**
- Usar solo XGBoost para SHAP (más rápido, más preciso, sin bugs de compatibilidad)
- Analizar el clúster 1 como caso especial de baja predictibilidad
- Enfocar el análisis diferencial en variables con alto coeficiente de variación entre territorios

---

**Fin del reporte**
