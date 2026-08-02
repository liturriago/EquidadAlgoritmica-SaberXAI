# Reporte — Clustering socioeconómico mediante Análisis de Clases Latentes (LCA)

**Notebook:** `08_clustering_socioeconomico.ipynb`
**Dataset:** Saber 11 (ICFES) — `v2_datos_limpios.csv`, 4 821 037 registros
**Algoritmo final:** Latent Class Analysis (LCA) vía `StepMix`
**K seleccionado:** 7 clases latentes

> Nota sobre el algoritmo: el encabezado del notebook menciona K-Modes como referencia conceptual, pero el modelo efectivamente ajustado y usado para asignar a la población es un **Análisis de Clases Latentes (LCA)** implementado con la librería `StepMix`. Este reporte documenta el modelo LCA, que es el que produce las clases finales.

---

## 1. Variables seleccionadas y por qué

El objetivo del clustering es agrupar a los estudiantes según su **perfil socioeconómico y de contexto escolar**, usando únicamente variables **categóricas y ordinales**. Se seleccionaron las 20 covariables que describen el contexto del estudiante, su familia/hogar y su colegio, dejando fuera la variable objetivo y las de cardinalidad inmanejable.

**Las 20 variables seleccionadas:**

| Variable | Tipo | Descripción |
|---|---|---|
| `cole_area_ubicacion` | binaria | Rural / Urbano |
| `cole_bilingue` | binaria | Colegio bilingüe (S/N) |
| `cole_calendario` | nominal (3) | Calendario A / B / Otro |
| `cole_caracter` | nominal | Carácter académico del colegio |
| `cole_depto_ubicacion` | nominal (33) | Departamento del colegio |
| `cole_genero` | nominal | Mixto / masculino / femenino |
| `cole_jornada` | nominal (6) | Jornada escolar |
| `cole_naturaleza` | binaria | Oficial / No Oficial |
| `cole_sede_principal` | binaria | Es sede principal (S/N) |
| `estu_genero` | binaria | Género del estudiante (M/F) |
| `fami_estratovivienda` | ordinal | Proxy oficial de nivel socioeconómico en Colombia |
| `fami_educacionmadre` | ordinal | Nivel educativo de la madre |
| `fami_educacionpadre` | ordinal | Nivel educativo del padre |
| `fami_tieneinternet` | binaria | Acceso a internet |
| `fami_tienecomputador` | binaria | Tiene computador |
| `fami_tieneautomovil` | binaria | Tiene automóvil |
| `fami_tienelavadora` | binaria | Tiene lavadora |
| `fami_tieneserviciotv` | binaria | Tiene servicio de TV |
| `fami_cuartoshogar` | ordinal | Cuartos en el hogar |
| `fami_personashogar` | ordinal | Personas en el hogar |

**Por qué estas:** describen tres dimensiones del contexto del estudiante que la literatura asocia con desigualdad educativa: (a) el **hogar** (estrato, educación de los padres, bienes durables como computador, internet, automóvil, lavadora, TV — proxies clásicos de capacidad económica), (b) la **estructura del hogar** (cuartos y personas, indicadores de hacinamiento), y (c) el **colegio** (naturaleza oficial/privada, área rural/urbana, jornada, carácter, calendario, bilingüismo). El conjunto captura el entorno que rodea al estudiante sin usar su resultado.

**Variables excluidas y por qué:**

| Variable | Motivo de exclusión |
|---|---|
| `punt_global` | Es el **objetivo** del proyecto. Se deja fuera del modelo para poder usarlo después como **validación externa**: si las clases (construidas sin el puntaje) ordenan el puntaje, el agrupamiento captura estructura real. |
| `cole_mcpio_ubicacion` | Municipio crudo de **alta cardinalidad** (~1 000 niveles). Introduce ruido y dispersa la señal; el departamento ya aporta la dimensión geográfica. |
| `anio`, `semestre` | Variables temporales numéricas, no son rasgos de perfil socioeconómico. |

**Limpieza previa relevante:** se normalizaron strings (strip + title case), se convirtieron a `NaN` los valores sin posición ordinal real en educación de padres (`"No Sabe"`, `"No Aplica"`), y se aplicó `dropna` sobre las 20 features (LCA no admite faltantes). Se conservaron **4 445 422 de 4 821 037 registros (92.2 %)**. Además, `fami_cuartoshogar` y `fami_personashogar` se re-mapearon para unificar categorías mezcladas (valores individuales con rangos agrupados) y dejar una escala ordinal limpia.

---

## 2. ¿Qué es LCA y por qué se escogió?

### Qué es

El **Análisis de Clases Latentes (Latent Class Analysis, LCA)** es un modelo estadístico de **mixtura finita** para datos **categóricos**. Parte de un supuesto: la población no es homogénea, sino que está compuesta por un número finito **K** de subgrupos ocultos ("clases latentes") que no observamos directamente. Cada estudiante pertenece a una de esas clases, y su clase determina la probabilidad de responder cada variable de cierta manera.

Dicho de otro modo: en lugar de medir "distancia" entre individuos (como K-Means o K-Modes), LCA propone un **modelo generativo probabilístico** — describe cómo cada clase latente "genera" los patrones de respuesta observados, y luego estima por máxima verosimilitud qué clases y qué probabilidades reproducen mejor los datos reales.

### Por qué se escogió este modelo (frente a K-Modes / K-Means)

1. **Es nativo para datos categóricos.** Las 20 variables son binarias, nominales y ordinales; no hay magnitudes continuas. K-Means es inadecuado (asume distancias euclidianas sobre números continuos). LCA modela directamente la probabilidad de cada categoría.

2. **Es probabilístico, no de "frontera dura".** A diferencia de K-Modes (que asigna cada punto al centroide más cercano de forma rígida), LCA entrega para cada estudiante una **probabilidad posterior de pertenecer a cada clase**. Esto permite medir la **confianza** de cada asignación (aquí, probabilidad posterior media de **0.993** — asignaciones casi deterministas) y diagnosticar solapamiento entre clases.

3. **Tiene criterios de selección de K bien fundamentados.** Al ser un modelo de verosimilitud, se pueden usar criterios de información (BIC, AIC, aBIC) y entropía para elegir el número de clases de forma objetiva, en lugar de heurísticas visuales.

4. **Respeta los tipos de variable.** Cada bloque de variables se declara con su distribución apropiada (ver paso a paso), algo que K-Modes no distingue (trata todo como nominal con distancia de Hamming).

---

## 3. Cómo funciona LCA — paso a paso

### El modelo formal (intuición)

LCA estima dos conjuntos de parámetros:

- **Probabilidades de clase** π₁, …, π_K: qué fracción de la población pertenece a cada clase latente (deben sumar 1).
- **Probabilidades condicionales** dentro de cada clase: para cada variable, la probabilidad de cada respuesta *dado que* el estudiante pertenece a la clase k.

La probabilidad de observar el patrón de respuestas de un estudiante es la **suma ponderada** sobre las K clases: cada clase "vota" con su probabilidad π_k multiplicada por lo bien que esa clase explica las respuestas del estudiante.

#### Verosimilitud del patrón de respuestas

Para un estudiante con respuestas $x_i = (x_{i1}, \dots, x_{iJ})$ en las $J$ variables, la probabilidad del patrón es la mixtura sobre las $K$ clases:

$$P(x_i \mid \theta) \;=\; \sum_{k=1}^{K} \pi_k \prod_{j=1}^{J} P(x_{ij} \mid \text{clase}=k)$$

donde:
- $\pi_k$ es la **probabilidad a priori** de la clase $k$, con $\sum_{k=1}^{K}\pi_k = 1$.
- $\prod_{j} P(x_{ij}\mid k)$ es el **supuesto de independencia condicional local**: *dentro* de una clase, las variables son independientes entre sí, así que la probabilidad conjunta es el producto de las marginales por variable.
- La forma de $P(x_{ij}\mid k)$ depende del tipo declarado de la variable $j$: **Bernoulli** para las binarias, $P(x_{ij}\mid k)=p_{jk}^{x_{ij}}(1-p_{jk})^{1-x_{ij}}$; y **gaussiana diagonal** para nominales/ordinales codificadas como enteros, $P(x_{ij}\mid k)=\mathcal{N}(x_{ij};\,\mu_{jk},\sigma_{jk}^2)$.

#### Función objetivo: log-verosimilitud (la "función de pérdida")

El modelo no minimiza un error: **maximiza la log-verosimilitud** de los datos (equivalentemente, minimiza la log-verosimilitud negativa). Sobre $n$ estudiantes:

$$\mathrm{LL}(\theta) \;=\; \sum_{i=1}^{n} \log P(x_i \mid \theta) \;=\; \sum_{i=1}^{n} \log\!\left( \sum_{k=1}^{K} \pi_k \prod_{j=1}^{J} P(x_{ij}\mid k) \right)$$

En el notebook esto aparece como `avg_ll = mod.score(...)` (log-verosimilitud media por observación) y `ll = avg_ll * n_eff` (la $\mathrm{LL}$ total). Todos los criterios de la sección 4 se construyen a partir de esta $\mathrm{LL}$.

### El algoritmo de estimación (EM — Expectation-Maximization)

LCA no observa las clases, así que las estima de forma iterativa:

1. **Inicialización.** Se parte de valores iniciales (aquí con `n_init=5`: cinco arranques distintos, conservando el de mayor verosimilitud, para evitar mínimos locales).
2. **Paso E (Expectation).** Con los parámetros actuales, se calcula la **probabilidad posterior** de que cada estudiante pertenezca a cada clase, vía regla de Bayes:

   $$\tau_{ik} \;=\; P(\text{clase}=k \mid x_i) \;=\; \frac{\pi_k \prod_{j} P(x_{ij}\mid k)}{\sum_{m=1}^{K} \pi_m \prod_{j} P(x_{ij}\mid m)}$$

   Estas $\tau_{ik}$ son las "responsabilidades": cuánto le pertenece el estudiante $i$ a la clase $k$ (suman 1 sobre $k$). La asignación final de cada estudiante es $\hat{k}_i = \arg\max_k \tau_{ik}$ (clase modal), y `prob_max_lca` $= \max_k \tau_{ik}$.
3. **Paso M (Maximization).** Con esas probabilidades como pesos, se **re-estiman** las probabilidades de clase, $\pi_k = \frac{1}{n}\sum_i \tau_{ik}$, y las condicionales, para maximizar la verosimilitud.
4. **Iteración.** Se repiten E y M hasta que la verosimilitud converge (deja de mejorar).

### Cómo se aplicó concretamente en este notebook

**a) Codificación por tipo de variable.** Cada bloque se declaró con su distribución de medición correspondiente en `StepMix`:

| Bloque | Modelo de medición | Nº variables |
|---|---|---|
| Binarias | `bernoulli` | 10 |
| Nominales | `gaussian_diag` (entero codificado) | 5 |
| Ordinales | `gaussian_diag` (entero codificado) | 5 |

Se codificó cada grupo con `OrdinalEncoder` (las ordinales con su orden natural explícito), produciendo una matriz `X_lca` de **(4 445 422 × 20)** valores enteros 0-indexados.

**b) Estrategia de ajuste en dos pasos (por escala).** Con 4,4 M de filas, ajustar el EM sobre toda la población es costoso. El notebook intenta primero un atajo:

- **Paso A — patrones únicos ponderados.** Si el número de combinaciones distintas de respuestas fuera ≤ 500 000, se ajustaría el LCA ponderado sobre esos patrones (exacto y rápido). Pero resultaron **2 691 488 patrones únicos** (60.5 % de las filas son combinaciones distintas) — demasiados.
- **Paso B — muestra estratificada.** Al superar el umbral, se tomó una **muestra estratificada de 299 992 filas**, estratificando por `cole_area_ubicacion × fami_estratovivienda` (14 celdas) con un piso de 500 filas por celda, para no perder subgrupos minoritarios (p. ej. Rural × Estrato 6).

**c) Selección de K** (paso C, ver sección siguiente): se ajustó LCA para K = 1…10 sobre la muestra.

**d) Modelo final.** Con K = 7 se reutilizó el mejor modelo del barrido.

**e) Asignación de toda la población.** Un único **paso E** sobre los 4,4 M de registros calcula las probabilidades posteriores y asigna a cada estudiante su **clase modal** (la más probable). Resultado guardado en `saber11_lca_clases.parquet`.

---

## 4. Métricas para escoger el número de clusters

LCA permite elegir K con criterios cuantitativos. Todos parten de la log-verosimilitud $\mathrm{LL}$ (sección 3) y le restan una **penalización por complejidad**, donde $p$ = número de parámetros del modelo (`n_par`) y $n$ = número de observaciones efectivas (`n_eff`). En el notebook se usaron cuatro criterios, que se complementan:

**BIC (Bayesian Information Criterion).** Penaliza la complejidad con $\log n$ por parámetro; es la penalización más fuerte de las tres. **Menor es mejor.** Se busca el "codo".

$$\mathrm{BIC} \;=\; -2\,\mathrm{LL} \;+\; p\,\log(n)$$

**AIC (Akaike Information Criterion).** Penaliza con una constante (2) por parámetro: penaliza menos que el BIC, por eso tiende a favorecer más clases. Se reporta como referencia.

$$\mathrm{AIC} \;=\; -2\,\mathrm{LL} \;+\; 2\,p$$

**aBIC (sample-size adjusted BIC).** Variante del BIC que reemplaza $n$ por el tamaño muestral "ajustado" $n^* = (n+2)/24$; suaviza la penalización y es la recomendada para modelos de clases latentes.

$$\mathrm{aBIC} \;=\; -2\,\mathrm{LL} \;+\; p\,\log\!\left(\frac{n+2}{24}\right)$$

**Entropía relativa de clasificación.** Mide qué tan **netas** son las asignaciones a partir de las posteriores $\tau_{ik}$ del E-step. Primero la entropía total de las asignaciones,

$$E \;=\; -\sum_{i=1}^{n}\sum_{k=1}^{K} \tau_{ik}\,\log \tau_{ik},$$

y luego se normaliza al rango $[0,1]$ contra el caso de máxima incertidumbre ($n\log K$):

$$E_{\text{rel}} \;=\; 1 \;-\; \frac{E}{n\,\log K}$$

$E_{\text{rel}} = 1$ ⇒ cada estudiante pertenece a una sola clase con certeza total (las $\tau_{ik}$ son 0/1); valores bajos ⇒ clases solapadas/ambiguas. **Mayor es mejor.** (En el código: `entropia = 1 - H / (n_eff * np.log(k))`.)

**Tamaño de la clase mínima (%).** Salvaguarda contra el **sobreajuste**. Con los tamaños relativos de clase $\hat{\pi}_k = \frac{1}{n}\sum_i \tau_{ik}$:

$$\text{clase mín.} \;=\; 100 \times \min_{k}\,\hat{\pi}_k$$

Si la clase más pequeña cae por debajo de ~1 %, suele tratarse de un artefacto del EM (un grupito espurio) y no de un subgrupo real de la población.

La combinación es clave: BIC/aBIC/AIC dicen *cuánta estructura* captura el modelo; la entropía dice *si las clases son distinguibles*; la clase mínima dice *si son reales o ruido*.

---

## 5. Cómo se escogió el número de clusters — análisis

**Tabla del barrido K = 1…10** (sobre la muestra estratificada de 299 992 filas):

| K | BIC | Δ BIC | Entropía | Clase mín. (%) |
|---|---|---|---|---|
| 5 | 3 423 087 | — | 0.956 | 2.8 |
| 6 | 1 904 866 | −1 518 221 | 0.991 | 2.8 |
| **7** | **1 267 824** | **−637 042** | **0.992** | **2.8** |
| 8 | 1 222 614 | −45 210 | 0.974 | 2.8 |
| 9 | 969 791 | −252 823 | 0.999 | ⚠️ 0.2 |
| 10 | 1 036 830 | +67 039 ↑ | 0.992 | ⚠️ 0.2 |

![Selección de K en el modelo LCA](reporte_lca_fig/seleccion_k.png)

*Figura 1 — Tres paneles de la decisión. Izquierda: BIC y aBIC (prácticamente superpuestos) cayendo abruptamente y aplanándose en K = 7. Centro: entropía relativa, en su techo (~0.99) desde K = 6. Derecha: tamaño de la clase mínima, que se desploma por debajo del umbral del 1 % a partir de K = 9.*

**Decisión: K = 7**, sostenida por cuatro argumentos convergentes:

1. **Codo del BIC en K = 7.** De K = 6 a K = 7 el BIC cae **637 042 puntos**; de K = 7 a K = 8 baja solo **45 210** (un 93 % menos de ganancia). El cambio brusco de pendiente indica que la octava clase ya no recupera estructura latente nueva, solo subdivide clases existentes.

2. **Entropía en su techo práctico (0.992).** Una entropía tan cercana a 1 implica asignación casi determinista: cada estudiante cae en una única clase con alta certeza. K = 7 ya alcanza ese máximo; K = 8 y K = 9 no lo mejoran de forma sustancial.

3. **Clases mínimas sanas (≥ 2.8 %).** A partir de K = 9 aparece una clase con apenas **0.2 %** de la muestra (~600 individuos de 300 000): señal de sobreajuste del EM, no de un subgrupo poblacional real. K = 7 mantiene todas las clases por encima del 2.8 %.

4. **BIC no monótono más allá de K = 9.** El BIC **sube** de K = 9 a K = 10 (969 791 → 1 036 830): el modelo añade parámetros más rápido de lo que gana verosimilitud — evidencia de inestabilidad y de que K = 9 es probablemente un mínimo local del EM, no estructura real.

**Conclusión:** K = 7 es el punto donde el modelo gana máxima información por parámetro adicional, mantiene clases interpretables y estadísticamente robustas, y la entropía ya no mejora al crecer K.

---

## 6. Qué evidencian las imágenes de cluster

### 6.1 Perfil de las 7 clases (validación con el puntaje)

`punt_global` **no entró al modelo**, por lo que sirve de validación externa. Perfil resumido (moda de cada variable + mediana del puntaje):

| Clase | % pobl. | Med. puntaje | Estrato | Educ. padres | Internet/PC | Auto | Colegio |
|---|---|---|---|---|---|---|---|
| **5** | 5.2 | **229** | Estrato 1 | Primaria incompleta | **No / No** | No | Oficial |
| **3** | 18.7 | **244** | Estrato 2 | Secundaria completa | Sí / Sí | No | Oficial |
| **0** | 47.2 | 252 | Estrato 2 | Secundaria completa | Sí / Sí | No | Oficial |
| **2** | 16.4 | 252 | Estrato 1 | Secundaria completa | Sí / Sí | No | Oficial |
| **4** | 5.8 | 255 | Estrato 2 | Secundaria completa | Sí / Sí | No | Oficial |
| **6** | 3.9 | **289** | Estrato 3 | Educ. profesional completa | Sí / Sí | No | Oficial |
| **1** | 2.9 | **307** | Estrato 3 | **Educ. profesional completa** | Sí / Sí | **Sí** | **No Oficial** |

Lo que se evidencia: las clases forman un **gradiente socioeconómico** que ordena el puntaje sin haberlo visto. En un extremo, la **clase 5** (estrato 1, padres con primaria incompleta, **sin internet ni computador**) tiene la mediana más baja (229). En el otro, la **clase 1** (estrato 3, padres profesionales, **con automóvil y colegio no oficial**) tiene la más alta (307). La clase 6 (profesionales, estrato 3, oficial) queda en segundo lugar (289). La mayoría de la población (clases 0, 2, 3, 4 ≈ 88 %) se concentra en un bloque medio homogéneo (estrato 1–2, secundaria completa, con conectividad básica), con puntajes medianos similares (244–255).

### 6.2 Boxplot del puntaje por clase

![punt_global por clase LCA](reporte_lca_fig/boxplot_puntaje_clase.png)

*Figura 2 — Distribución del puntaje global por clase, ordenada por mediana (muestra de 50 000).*

Lo que se identifica: las medianas suben de forma **monótona** de la clase 5 a la 1, confirmando el gradiente. Las clases que más se separan del resto son los **extremos** (5 abajo, y 6/1 arriba), mientras que el bloque central (3, 0, 2, 4) tiene cajas muy solapadas — son socioeconómicamente parecidos y sus puntajes también. La clase 1 (la más aventajada) muestra además la **mayor dispersión** (caja y bigotes más amplios): es un grupo más heterogéneo. La validación externa funciona: una agrupación construida solo con contexto **predice la dirección** del resultado académico.

### 6.3 Proyección UMAP

![UMAP de las 20 variables socioeconómicas](reporte_lca_fig/umap_clases.png)

*Figura 3 — UMAP (métrica de Hamming, 20 000 individuos). Izquierda: coloreado por clase LCA. Derecha: mismo embedding coloreado por puntaje global (validación externa).*

**Por qué UMAP y no PCA:** las variables son categóricas/ordinales y PCA asume relaciones lineales entre números continuos (no tiene sentido tratar "Estrato 1 → 2" igual que "Estrato 5 → 6"). UMAP con **distancia de Hamming** (cuenta cuántos atributos difieren entre dos estudiantes) usa la misma lógica de similitud que el LCA, y preserva tanto la estructura local como la global.

Lo que se evidencia:

- **Panel izquierdo (por clase):** las 7 clases ocupan **regiones coherentes y contiguas** del embedding — los colores no están mezclados al azar, sino que forman zonas reconocibles. Esto confirma que las clases del LCA corresponden a regiones reales del espacio de respuestas, no a particiones arbitrarias.
- **Panel derecho (por puntaje):** el color del puntaje varía de forma **gradual y ordenada** a lo largo del mapa, alineado con la disposición de las clases. La transición suave (no bloques abruptos) sugiere que el contexto socioeconómico es un **continuo** más que categorías totalmente disjuntas: las clases son cortes útiles dentro de un gradiente, y la zona del puntaje alto coincide con la región de las clases aventajadas (1 y 6).

---

## 7. Síntesis

Se agrupó a 4,4 M de estudiantes con un **LCA de 7 clases** sobre 20 variables socioeconómicas y de contexto escolar, excluyendo el puntaje para usarlo como validación. LCA se eligió por ser probabilístico y nativo para datos categóricos, con criterios objetivos de selección de K. El número de clases (7) se fijó por la convergencia de cuatro señales: codo del BIC, entropía en su techo (0.992), clases mínimas robustas (≥ 2.8 %) y estabilidad del BIC. Las tres visualizaciones confirman que las clases (a) ordenan monótonamente un puntaje que nunca vieron, (b) ocupan regiones coherentes del espacio de respuestas y (c) trazan un gradiente socioeconómico continuo desde el grupo más vulnerable (clase 5: sin conectividad, padres con primaria incompleta) hasta el más aventajado (clase 1: estrato alto, padres profesionales, colegio privado).
