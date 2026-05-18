# Documento de Contexto del Proyecto — Para Análisis Externo

> **Propósito de este documento:** Proveer contexto completo y preciso del proyecto de investigación *"Equidad Algorítmica y Territorios Funcionales: Adaptación de Dominio y XAI en Pruebas Saber"* para que un modelo de lenguaje pueda analizar una tesis de doctorado externa e identificar cómo los conceptos, métodos y marcos teóricos de dicha tesis son aplicables a este proyecto.

---

## 1. Título y Naturaleza del Proyecto

**Título oficial:** Equidad Algorítmica y Territorios Funcionales: Adaptación de Dominio y XAI en Pruebas Saber

**Tipo:** Investigación cuantitativa predictiva, convocatoria estudiantil ICFES 2026.

**Institución:** Instituto Colombiano para la Evaluación de la Educación (ICFES).

**En una frase:** Se aplica deep learning con técnicas de Adaptación de Dominio (DANN) e Inteligencia Artificial Explicable (XAI / SHAP) sobre los microdatos censales del examen Saber 11 para predecir el puntaje global de los estudiantes y, sobre todo, para cuantificar cómo las brechas de equidad —estrato socioeconómico, brecha digital, género, etnia, ruralidad— interactúan y se amplifican entre sí.

---

## 2. Problema que Resuelve

### El vacío que existe hoy

Los modelos estadísticos tradicionales que analizan desigualdad educativa en Colombia asumen que los efectos de las variables (estrato, conectividad, género, etnia) son **aditivos e independientes**. Esto produce dos errores graves:

1. **Sesgo urbanocéntrico (Covariate Shift / Domain Shift):** El dataset Saber 11 tiene ~87% de estudiantes urbanos y ~13% rurales. Un modelo entrenado sobre la población total aprende la distribución urbana como "la norma" y colapsa empíricamente al evaluar estudiantes rurales, generando predicciones con mayor error absoluto precisamente para las poblaciones más vulnerables.

2. **Opacidad algorítmica:** Los modelos que sí corrigen el sesgo geográfico (como las Redes Adversariales de Dominio) operan como "cajas negras" inescrutables, lo que impide comprender las causas estructurales de la vulnerabilidad y hace imposible diseñar políticas públicas basadas en sus hallazgos.

**El proyecto resuelve ambos problemas simultáneamente**, siendo el primero en Colombia en hacerlo sobre datos tabulares masivos del sistema educativo.

---

## 3. Pregunta de Investigación

**Pregunta principal:**
¿De qué manera la aplicación de técnicas de Adaptación de Dominio e Inteligencia Artificial Explicable (XAI) en modelos predictivos permite cuantificar cómo las dimensiones del enfoque diferencial (estrato socioeconómico, brecha digital, género y pertenencia étnico-racial) interactúan con los territorios funcionales para configurar brechas de rendimiento en las pruebas Saber 11?

**Preguntas secundarias:**
1. ¿En qué medida los modelos de aprendizaje automático tradicionales sufren de cambios de distribución (Domain Shift) al evaluar factores asociados al aprendizaje en territorios rurales dispersos frente a núcleos metropolitanos?
2. ¿De qué forma la implementación de algoritmos con Adaptación de Dominio permite mitigar el sesgo urbanocéntrico y garantizar la equidad algorítmica al predecir el desempeño de poblaciones vulnerables?
3. ¿Cómo la extracción de métricas de explicabilidad local (valores SHAP) logra operacionalizar el concepto de interseccionalidad, desentrañando el peso multiplicativo de las carencias socioeconómicas y tecnológicas según el cruce identitario y geográfico del estudiante?

---

## 4. Objetivos

### Objetivo General
Analizar, mediante técnicas de Adaptación de Dominio e Inteligencia Artificial Explicable (XAI), cómo las dimensiones del enfoque diferencial (estrato socioeconómico, brecha digital, género y etnia) interactúan con los territorios funcionales para configurar brechas de rendimiento en las pruebas Saber 11, proveyendo evidencia algorítmica auditable para la focalización de políticas educativas.

### Objetivos Específicos
1. **Cuantificar** la magnitud del sesgo distribucional (Domain Shift) presente en los modelos de aprendizaje automático tradicionales al predecir el rendimiento académico de poblaciones en territorios rurales aislados frente a centros metropolitanos.
2. **Implementar** arquitecturas de Adaptación de Dominio Profundo (Deep DA) para extraer representaciones latentes invariantes a la geografía, mitigando el sesgo urbanocéntrico y garantizando la equidad algorítmica (Fairness).
3. **Operativizar** la "interseccionalidad computacional" mediante métricas de explicabilidad local (valores SHAP), aislando matemáticamente el peso multiplicativo y no lineal que ejercen las matrices de exclusión cruzadas (bajo estrato + etnia + carencia de conectividad) sobre el desempeño estudiantil.

---

## 5. Marco Conceptual Clave

Estos son los conceptos teóricos sobre los cuales se construye el proyecto. Cualquier tesis que trabaje alguno de ellos es potencialmente relevante:

### 5.1 Territorios Funcionales
Se abandona la dicotomía urbano/rural estática. La "ruralidad" se entiende de forma dinámica: los territorios funcionales reconocen que las regiones periféricas poseen dinámicas de exclusión radicalmente distintas a los núcleos metropolitanos, incluyendo acceso a infraestructura educativa, distancias, oferta docente y conectividad. El área de ubicación del **colegio** (no la residencia del estudiante) define el dominio geográfico en este proyecto, porque determina el contexto institucional donde ocurre el aprendizaje.

### 5.2 Interseccionalidad
La desigualdad no es aditiva. Pertenecer simultáneamente a múltiples grupos vulnerables (estrato bajo + ruralidad + sin internet + minoría étnica) genera una desventaja multiplicativa, no simplemente la suma de cada desventaja por separado. El proyecto busca cuantificar matemáticamente este efecto multiplicativo a nivel individual con SHAP.

### 5.3 Domain Shift / Covariate Shift
Fenómeno estadístico que ocurre cuando la distribución de las variables de entrada (X) cambia entre el dominio fuente (donde se entrena el modelo: mayoritariamente urbano) y el dominio objetivo (donde se aplica: rural disperso). El modelo no es consciente de este cambio y produce predicciones con sesgo sistemático hacia las características de la población mayoritaria.

### 5.4 Redes Neuronales Adversariales de Dominio (DANN)
Arquitectura de deep learning propuesta por Ganin et al. (2016). Consiste en tres componentes:
- Un **extractor de características** compartido
- Un **predictor de tarea** (predice el puntaje global)
- Un **discriminador de dominio** subsidiario (intenta predecir si el estudiante es urbano o rural)

Durante el entrenamiento, se aplica una **inversión de gradiente**: el extractor es penalizado si el discriminador logra identificar el origen geográfico. Esto fuerza al modelo a aprender representaciones latentes **invariantes al territorio**, extrayendo solo las características académicamente relevantes que son válidas independientemente de si el estudiante viene de Bogotá o de una vereda en el Chocó.

### 5.5 XAI — Inteligencia Artificial Explicable con SHAP
Los valores SHAP (SHapley Additive exPlanations, Lundberg & Lee 2017) son una técnica de explicabilidad basada en la teoría de juegos cooperativos. Asignan a cada variable predictora su contribución marginal exacta sobre la predicción individual de un estudiante. En este proyecto se usan de forma condicional (respetando las correlaciones reales entre variables socioeconómicas, que son altas) y con *feature grouping* (agrupando variables correlacionadas como las `fami_*` para evitar valores SHAP inestables por multicolinealidad).

El resultado son **diagramas de enjambre multifactoriales** que revelan el peso multiplicativo de las vulnerabilidades cruzadas, y **matrices de vulnerabilidad interseccional** que muestran qué combinaciones de carencias producen los mayores efectos negativos sobre el puntaje.

---

## 6. Datos: Dataset ICFES Saber 11

### 6.1 Descripción General
- **Fuente:** FTP DataIcfes — repositorio oficial del ICFES
- **Cobertura temporal:** 21 periodos semestrales, de 2014-2 a 2024-2 (se excluye 2014-1 por incompatibilidad de esquema: 138 columnas vs. 57–85 del resto)
- **Registros:** 4.821.037 filas limpias (de 7.239.505 originales; 66.59% conservado tras eliminar filas con nulos)
- **Segmentación:** 86.82% urbano (4.185.434 filas) — 13.18% rural (635.603 filas)

### 6.2 Variable Objetivo (Target)
`punt_global`: puntaje global del examen Saber 11. Se excluyen los sub-puntajes por área (matemáticas, lectura crítica, ciencias naturales, sociales, inglés) porque son componentes directos del target y generarían data leakage.

### 6.3 Variables Predictoras (28 features)

| Grupo | Variables |
|-------|-----------|
| **Colegio** | `cole_area_ubicacion`, `cole_bilingue`, `cole_calendario`, `cole_caracter`, `cole_depto_ubicacion`, `cole_genero`, `cole_jornada`, `cole_mcpio_ubicacion`, `cole_naturaleza`, `cole_sede_principal` |
| **Estudiante** | `estu_depto_reside`, `estu_fechanacimiento`, `estu_genero`, `estu_grado`, `estu_tipodocumento` |
| **Familia / Socioeconómico** | `fami_estratovivienda`, `fami_educacionmadre`, `fami_educacionpadre`, `fami_cuartoshogar`, `fami_personashogar`, `fami_tienecomputador`, `fami_tieneinternet`, `fami_tienelavadora`, `fami_tieneautomovil`, `fami_tieneserviciotv` |
| **Temporal** | `anio`, `semestre` |

**Notas importantes sobre variables:**
- Se excluyen los índices compuestos INSE y NSE (calculados por el ICFES a partir de las variables `fami_*`) porque se prefieren las variables desagregadas para que SHAP pueda identificar qué dimensión específica del nivel socioeconómico tiene mayor impacto.
- `estu_grado` se conserva aunque el 88% son grado 11: el 11.91% restante son **validantes de bachillerato (grado 26)**, adultos con perfiles socioeconómicos y de desempeño sistemáticamente distintos, relevantes para equidad.
- `estu_tipodocumento` actúa como proxy de estatus migratorio (CE, pasaporte = población con condiciones de acceso diferenciadas).

### 6.4 Decisiones de Preparación de Datos
- **Lectura con `dtype=str`** para evitar coerciones silenciosas entre periodos con esquemas distintos.
- **Normalización geográfica** (`strip_accents`): el ICFES registra "Bogotá" y "Bogota" como valores distintos según el periodo; la normalización Unicode reduce `cole_mcpio_ubicacion` de 1.396 a 1.081 valores únicos.
- **Descomposición de `periodo`** en `anio` (int) + `semestre` (str) para capturar tendencia temporal continua y efecto semestral por separado.
- **Filtro de edad 14–60** solo en análisis exploratorio (no en entrenamiento).
- El **dominio fuente** es `cole_area_ubicacion = Urbano`; el **dominio objetivo** es `cole_area_ubicacion = Rural`.

---

## 7. Metodología — Tres Fases Computacionales

### Fase 1 — Cuantificación del Domain Shift (Semanas 1–8)
1. Preprocesamiento para alta cardinalidad y desbalanceo de clases.
2. Entrenamiento de modelo no lineal base (XGBoost o Perceptrón Multicapa) sobre la población total.
3. Evaluación de la brecha de error absoluto: probar el modelo en dominio fuente (urbano) vs. dominio objetivo (rural disperso).
4. Cuantificación matemática de la degradación por sesgo distribucional → esto responde el OE1.

### Fase 2 — Adaptación de Dominio Profundo / DANN (Semanas 9–14)
1. Implementación de la arquitectura DANN con inversión de gradiente.
2. El extractor de características es penalizado si el discriminador de dominio logra identificar el origen geográfico.
3. Ajuste hiperparamétrico dinámico del factor de penalización adversarial.
4. Experimentación con Adaptación de Dominio Condicional (alinea distribuciones condicionadas por clase de rendimiento) para evitar transferencia negativa → esto responde el OE2.

### Fase 3 — Interseccionalidad Computacional / XAI (Semanas 15–22)
1. Extracción de valores SHAP condicionales con feature grouping.
2. Análisis global (qué variables importan más en promedio) y local (cuánto pesa cada variable para un estudiante específico).
3. Generación de diagramas de enjambre multifactoriales por perfiles interseccionales.
4. Cuantificación matemática de las matrices de vulnerabilidad cruzada → esto responde el OE3.

---

## 8. Limitaciones Reconocidas

1. **Multicolinealidad sociodemográfica:** En Colombia, estrato, falta de internet y región están fuertemente correlacionadas. SHAP marginal puede generar escenarios sintéticos socialmente imposibles. **Solución:** SHAP condicional + feature grouping.
2. **Riesgo de transferencia negativa:** La DANN podría borrar características correlacionadas con el territorio pero legítimamente predictivas de competencias cognitivas. **Solución:** Adaptación de Dominio Condicional.
3. **Variables omitidas estructuralmente:** El dataset no captura salud mental, violencia intrafamiliar ni calidad pedagógica docente. **Solución:** Las conclusiones se enmarcan exclusivamente como determinantes estructurales de clase e infraestructura.

---

## 9. Entregables Formales

| Producto | Descripción | Semana |
|----------|-------------|--------|
| Repositorio open source | Pipeline end-to-end bajo principios MLOps | 25 |
| Working Paper | Documento científico con hallazgos completos | 26–28 |
| Policy Brief | Recomendaciones territoriales accionables para política pública | 26–28 |
| Socialización Icfes | Seminario de Investigación | 29–30 |
| Socialización UNAL | Seminario de Investigación | 29–30 |

---

## 10. Referencias Clave del Proyecto

- Ganin, Y. et al. (2016). Domain-adversarial training of neural networks. *Journal of Machine Learning Research*, 17(1), 2096–2030.
- Lundberg, S.M. & Lee, S.I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
- Sugiyama, M. et al. (2007). Covariate shift adaptation by importance weighted cross validation. *Journal of Machine Learning Research*, 8(5).
- Suaza-Medina, M. et al. (2024). A model for predicting academic performance on standardised tests for lagging regions based on machine learning and Shapley additive explanations. *Scientific Reports*, 14(1), 25306.
- Balogun, M. et al. (2025). Transfer Learning Applications in Predicting Student Success Across Different Educational Institutions: Mitigating Domain Shift through Deep Domain Adaptation.
- Delprato, M. (2025). Identifying the post-pandemic determinants of low performing students in Latin America through interpretable Machine Learning SHAP Values — Insights from PISA 2022. *arXiv:2509.24508*.
- Icfes. (2023). Guía teórico-metodológica para la transversalización del enfoque diferencial y el análisis interseccional. Bogotá.
- Fernández, J. et al. (2023). Enfoque territorial y análisis dinámico de la ruralidad: alcances y límites para el diseño de políticas de desarrollo rural innovadoras en América Latina y el Caribe.

---

## 11. Instrucción para el Modelo que Analiza la Tesis

Al recibir este documento junto con una tesis de doctorado, tu tarea es:

1. **Identificar** qué conceptos, métodos, marcos teóricos o hallazgos de la tesis son directamente aplicables a alguna de las tres fases metodológicas de este proyecto (cuantificación del Domain Shift, DANN, XAI/SHAP interseccional).
2. **Señalar** cómo la tesis puede fortalecer o refinar el marco conceptual del proyecto (territorios funcionales, interseccionalidad, equidad algorítmica).
3. **Proponer** de manera concreta cómo adaptar o integrar los aportes de la tesis en el pipeline técnico o en la argumentación teórica del proyecto.
4. **Indicar** si la tesis resuelve alguna de las tres limitaciones reconocidas del proyecto (multicolinealidad, transferencia negativa, variables omitidas).

El proyecto trabaja con datos tabulares de gran escala (~4.8 millones de registros), no con imágenes ni texto. Cualquier método propuesto debe ser viable en ese contexto computacional.
