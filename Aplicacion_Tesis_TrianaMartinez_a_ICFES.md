# Aplicación de la Tesis de Triana-Martinez (PhD) al Proyecto ICFES — Equidad Algorítmica y Territorios Funcionales

> **Documento base para presentación a tutores / director de tesis**
> **Tesis fuente:** Jenniffer Carolina Triana-Martinez, *"A deep learning approach with preserved interpretability to support multimodal remote sensing in precision agriculture with applications to the Tolima region"*, 218 pp., 2026.
> **Alcance considerado en este análisis:** Cap. 7 (Locality-Based Relevance Analysis), Cap. 8 (Attention-Based Interpretability) y Cap. 10 (Final Remarks). Los Caps. 1–6 (preliminares, remote sensing, GCE para múltiples anotadores y datos UAV) quedan fuera del alcance. El Cap. 9 (M-GCECDL) se analiza **únicamente como contraste metodológico** y se descarta como pieza aplicable: ver §1.3 para la justificación.
> **Proyecto destino:** *Equidad Algorítmica y Territorios Funcionales: Adaptación de Dominio y XAI en Pruebas Saber* — convocatoria estudiantil ICFES 2026.

---

## Resumen ejecutivo

La tesis de Triana-Martinez resuelve, para datos agrícolas multimodales del Tolima, dos problemas técnicos que el proyecto ICFES también enfrenta: (i) datos con **estructura no-lineal y no-estacionaria** que los métodos lineales clásicos (PCA, SVD-biplot) no capturan, y (ii) la **tensión interpretabilidad–precisión** en modelos profundos sobre datos tabulares. Aunque el dominio aplicado es distinto (sensores remotos sobre cultivos vs. microdatos educativos), **dos métodos de la tesis son técnicamente trasladables y conceptualmente honestos para ICFES: el UMAP-based Local Biplot (Cap. 7) y el TabNet-informed Local Biplot (Cap. 8)**. Ambos son herramientas de visualización e interpretabilidad agnósticas al dominio de aplicación, por lo que su traslado no requiere forzar analogías.

El **Multimodal GCECDL (M-GCECDL, Cap. 9) NO se traslada al proyecto**, aunque tiene una estructura sugerente: se argumenta en §1.3 que está diseñado para **fusión de canales sensoriales físicamente heterogéneos** (RGB + multiespectral + térmico), problema que ICFES no tiene. El sesgo urbano/rural del proyecto es un *covariate shift dentro de un mismo canal de adquisición* (el formulario Saber 11), y para eso la herramienta correcta es DANN/CDAN/Importance Weighting — no fusión multimodal con reliability gating.

Este documento (i) sintetiza los temas principales del trabajo base aplicables (Caps. 7, 8 y 10) con citas y fórmulas exactas, (ii) los mapea a las tres fases metodológicas del proyecto ICFES, (iii) propone integraciones concretas en el pipeline técnico, y (iv) justifica explícitamente por qué M-GCECDL **no** se incorpora.

---

## 1. Temas principales del trabajo base — Triana-Martinez (Caps. 7, 8 y 10)

La tesis enumera sus contribuciones en §10.2 *Conclusions* (p. 152–154). De ellas, **dos son técnicamente trasladables al proyecto ICFES sin forzar analogías**, citadas textualmente:

> *"Third, a Local Biplot framework informed by UMAP was developed to provide non-linear structure-preserving visualizations of high-dimensional data, revealing localized phenological patterns. Fourth, an Attention-Based Interpretability model using TabNet and Biplots was designed to disentangle key features across tabular and image-based modalities."* (§10.2, p. 153)

Estas dos contribuciones — el UMAP-based Local Biplot (Cap. 7) y el TabNet-informed Local Biplot (Cap. 8) — son **herramientas de visualización e interpretabilidad agnósticas al dominio**, por lo que su aplicabilidad a ICFES no depende de asumir que los datos educativos son sensorialmente análogos a los agrícolas. Las desarrollo en §1.1 y §1.2.

Una tercera contribución de la tesis, el M-GCECDL del Cap. 9, **se descarta como pieza aplicable** por un argumento conceptual desarrollado en §1.3 (su estructura está diseñada para canales sensoriales físicamente heterogéneos, mientras que ICFES recoge todas sus variables de un único canal — el formulario Saber 11).

### 1.1 UMAP-based Local Biplot — Cap. 7, §7.3

La tesis abandona el biplot lineal clásico (SVD sobre `X ∈ R^{N×P}`, Ec. 7-1) porque *"linear projection methods (e.g., PCA or SVD)..."* no capturan la **estructura no-lineal y no-estacionaria** típica de datos agronómicos (p. 79).

**Pipeline (Fig. 7-1, p. 81):**

1. **UMAP** sobre `X` → embedding `Z ∈ R^{N×2}` que preserva topología local vía un kNN-graph fuzzy. Los pesos del grafo de alta dimensión son (Ec. 7-3):

   ```
   w_{nn'} = exp(−max(0, (d(x_n, x_{n'}) − ρ_n) / σ_n))
   ```

   y los de baja dimensión son una distribución tipo Student-t (Ec. 7-4):

   ```
   w̃_{nn'} = (1 + α_UMAP · ‖z_n − z_{n'}‖²₂)^(−β_UMAP)
   ```

   El embedding óptimo minimiza la **fuzzy set cross-entropy** (Ec. 7-5):

   ```
   Z* = arg min Σ [w_{nn'} log(w_{nn'} / w̃_{nn'}) + (1 − w_{nn'}) log((1 − w_{nn'}) / (1 − w̃_{nn'}))]
   ```

2. **K-means en el espacio latente Z** (no en X original) — esto es el "key divergence from traditional methods" según p. 80 (Ec. 7-6):

   ```
   {Z̃_c*}_c = arg min Σ_c Σ_{z_n ∈ Z̃_c} ‖z_n − μ_c‖²₂
   ```

3. **SVD local por cluster:** para cada cluster `c`, sobre el subconjunto `X_c`:

   ```
   X_c ≈ Ũ_c S̃_c Ṽ_cᵀ,    B_c = Ṽ_c S̃_c^(0.5) ∈ R^{P×2}
   ```

4. **Transformación afín cluster-específica** para alinear la base local `B_c` con la geometría UMAP (Ec. 7-7):

   ```
   γ_c*, ν_c* = arg min ‖B̃_c(γ_c, ν_c) − B_c‖_F
   ```

5. **Ranking de variables por cluster** (Ec. 7-8):

   ```
   ψ_c = |B_c| · 1
   ```

**Cita clave (§7.6, p. 92):**
> *"Comparative analysis indicates that the proposed strategy effectively addresses the limitations of conventional SVD-based biplots regarding the identification and preservation of local topological structures... the method demonstrates superior capability in resolving latent sub-structures."*

**Por qué importa para ICFES:** el concepto de **territorios funcionales** del proyecto (§5.1 del contexto) afirma que la ruralidad no es una etiqueta binaria sino un continuo de exclusión multidimensional. El Local Biplot **operacionaliza geométricamente** ese constructo: en lugar de imponer la dicotomía `cole_area_ubicacion ∈ {Urbano, Rural}`, deja que los **clusters territoriales-funcionales emerjan empíricamente** del espacio latente UMAP.

---

### 1.2 TabNet-informed Local Biplot — Cap. 8

**Motivación textual (§8 introducción, p. 93):**
> *"the current chapter shifts focus toward predictive interpretability... A unified, explainable ML framework is proposed, integrating supervised prediction with unsupervised visualization to support transparent crop-trait modelling. The primary objective is to achieve a balance between high-fidelity prediction and the derivation of actionable, transparent insights."*

**Arquitectura TabNet (Arik & Pfister, 2021; §8.1, Ecs. 8-1 a 8-12):**

- Mecanismo de atención por pasos: en cada paso `i`, se aplica una máscara dispersa `M[i] ∈ R^{B×P}` (Ec. 8-2):

  ```
  M[i] = sparsemax(P[i−1] · h_i(a[i−1]))
  ```

  donde `sparsemax` (Martins & Astudillo, 2016) **fuerza selección rala** (sólo un subconjunto de features se activa por paso).

- La importancia global agregada del feature `j` (Ec. 8-12) es:

  ```
                       Σ_i Imp_b[i] · M_{b,j}[i]
  M_agg-b,j = ───────────────────────────────────
              Σ_j Σ_i Imp_b[i] · M_{b,j}[i]
  ```

- Loss conjunto (Ec. 8-11):

  ```
  L = L_task + λ_sparse · L_sparse
  ```

  con regularización de entropía sobre las máscaras (Ec. 8-10).

**Integración con Local Biplot (§8.3, p. 97):**
> *"While TabNet's output does not alter the inputs for UMAP or the biplot, the relevance scores are overlaid on the Local Biplot to evaluate the alignment between supervised and unsupervised perspectives. Specifically, the normalized values of M_agg are mapped to the colour intensity of variable arrows in the local biplots, highlighting features that are significant both visually and predictively."*

**Por qué importa para ICFES:** TabNet es una **alternativa nativa a SHAP** para datos tabulares con tres ventajas para el proyecto: (i) la sparsemax mitiga el problema de **multicolinealidad** (limitación 1 del proyecto, contexto §8) porque obliga al modelo a elegir un representante por grupo correlacionado; (ii) la importancia es **local por muestra** (igual que SHAP local), permitiendo cuantificar interseccionalidad; (iii) los embeddings de TabNet pueden alimentar directamente UMAP, evitando la pérdida de información del binning categórico que sufre SHAP marginal.

---

### 1.3 Por qué se descarta el M-GCECDL (Cap. 9)

El Capítulo 9 introduce el **Multimodal GCECDL (M-GCECDL)**, una arquitectura que asigna pesos de confianza por modalidad y por muestra para fusionar predicciones (encoders modulares + reliability gating + Product-of-Experts). En una primera versión de este documento se propuso adaptarlo al proyecto ICFES particionando las 28 features en cuatro "modalidades" (`cole_*`, `estu_*`, `fami_*`, temporal). **Esa propuesta se retira por inconsistencia conceptual.** La justificación es la siguiente:

**Lo que dice la tesis (cita literal, §9, p. 114):**
> *"input spaces naturally partitioned into multiple modalities... where heterogeneous sources of information (e.g., spectral, in-field, or topographic data) jointly describe each sample."*

La palabra operativa es **"heterogeneous sources of information"**. En la tesis las modalidades son **canales sensoriales físicamente distintos** (RGB + multiespectral + térmico, con diferentes principios de medición, diferentes longitudes de onda y diferentes fuentes de ruido) que capturan información complementaria sobre el mismo cultivo. El reliability gating `υ_n^(m)` aprende, por ejemplo, que el sensor térmico es poco fiable cuando la radiación solar es baja y que el RGB pierde discriminación cuando el dosel se densifica.

**Lo que ocurre en ICFES:**
- Todas las features —`cole_*`, `estu_*`, `fami_*`, temporales— provienen del **mismo formulario único** que el estudiante diligencia al inscribirse al Saber 11.
- Es **un único canal de adquisición**, no varios canales heterogéneos.
- Lo que el proyecto necesita resolver (sesgo urbano vs. rural) **no es heterogeneidad de canales**: es **covariate shift dentro del mismo canal** — la distribución de las mismas variables cambia entre poblaciones.

**Las dos cosas son problemas estructuralmente distintos:**

| Problema | Naturaleza | Herramienta correcta |
|---|---|---|
| Domain shift urbano vs. rural | Mismas features, distribuciones distintas (`P_urbano(X) ≠ P_rural(X)`) | DANN, CDAN, Importance Weighting |
| Modality reliability fusion | Diferentes features por canal, mismo sample | M-GCECDL |

Trasladar el M-GCECDL a ICFES tratando "grupos de features de un mismo cuestionario" como si fueran "canales sensoriales heterogéneos" sería:
1. **Conceptualmente forzado**: la tesis admite particiones "*disjoint or partially overlapping*" (§9, p. 114), pero la motivación física de canales independientes no se preserva.
2. **Funcionalmente desalineado**: el reliability gating no resolvería el problema que el proyecto necesita resolver (la diferencia de distribución urbano/rural), porque opera **entre grupos de features de la misma muestra**, no entre **muestras de distintas distribuciones**.
3. **Defensivamente débil frente al comité**: un reviewer informado señalaría la analogía forzada inmediatamente.

**Decisión:** el M-GCECDL queda excluido del pipeline ICFES. Para el sesgo urbano/rural se mantienen las herramientas clásicas de adaptación de dominio que el propio proyecto ya contempla (DANN como baseline y CDAN como refinamiento contra transferencia negativa). La tesis no ofrece aporte original sobre adaptación de dominio; ese no es su fuerte. Su fuerte es la **interpretabilidad geométrica y de atención**, y ese es exactamente lo que sobrevive a la traducción en §1.1 y §1.2.

---

## 2. Mapeo Tesis → Proyecto ICFES

A continuación presento la correspondencia entre las contribuciones de la tesis y las tres fases del pipeline ICFES (contexto §7).

### 2.1 Fase 1 — Cuantificación del Domain Shift (OE1)

| Aporte de la tesis | Aplicación en ICFES | Justificación textual |
|---|---|---|
| **UMAP + K-means en latent space** (§7.3) | Antes de evaluar el Domain Shift entre `cole_area_ubicacion = Urbano` vs. `Rural`, aplicar UMAP sobre las 28 features para descubrir si la dicotomía administrativa coincide con la **estructura topológica latente** de la población estudiantil. | *"instead of clustering the original features, our approach partitions the latent feature space"* (§7.3, p. 80). Probaría empíricamente si "urbano/rural" son realmente dos clusters o si existen 4–5 perfiles funcionales. |
| **Métricas de preservación de vecindario** (Trustworthiness, Continuity, Q_NX, R_NX, AUC, GNN gain, §8.4.1) | Cuantificar matemáticamente cuánta estructura urbana se "transfiere" a la rural y viceversa al hacer la división de dominio. Las métricas Q_NX(K) (Ec. 8-19) y AUC[R_NX(K)] (p. 99) son directamente reutilizables. | Estas métricas *"test local and global structure preservation at varying K-nearest neighbors"* (§8.4.1, p. 99) — exactamente lo que el proyecto necesita para el OE1. |

**Propuesta concreta:** en lugar de medir Domain Shift sólo como `MAE_rural − MAE_urbano` (formulación clásica), complementar con:
- Una **prueba topológica**: UMAP global de la población; medir la fracción de vecinos K-NN que cambian de etiqueta urbano↔rural en el embedding. Si la fracción es alta, el Domain Shift es topológicamente leve; si es baja, las dos distribuciones están geométricamente separadas en el espacio latente.
- Una **prueba de preservación**: entrenar UMAP sólo con urbanos, proyectar rurales como "out-of-sample" y medir Trustworthiness/Continuity.

---

### 2.2 Fase 2 — Adaptación de Dominio (OE2)

El proyecto contempla **DANN** (Ganin et al., 2016) con inversión de gradiente y, como refinamiento contra transferencia negativa, **Adaptación de Dominio Condicional (CDAN)**. **La tesis no aporta un método propio para adaptación de dominio** — su frente de innovación es la interpretabilidad, no la mitigación de covariate shift. Por eso esta fase **no recibe transferencia metodológica directa** desde la tesis.

Lo que sí pueden hacer Local Biplot y TabNet en esta fase es **diagnóstico complementario**, no sustitutivo:

| Pregunta diagnóstica para Fase 2 | Cómo la responde la tesis (Caps. 7 y 8) |
|---|---|
| ¿La frontera urbano/rural se manifiesta como **clusters geométricamente separados** en el espacio latente, o como un continuo? | Aplicar UMAP global; medir si los clusters K-means coinciden con `cole_area_ubicacion`. Si coinciden, el shift es categórico; si no, hay perfiles intermedios — argumento empírico a favor de **territorios funcionales** continuos. |
| ¿La DANN realmente reduce la separabilidad de dominio en el espacio latente? | Entrenar DANN, extraer embeddings antes/después, aplicar Local Biplot al embedding final. Si las flechas de `cole_area_ubicacion` se acortan tras la adaptación, hay evidencia visual de invariancia. |
| ¿Hay transferencia negativa? (¿La DANN borra señal predictiva legítima?) | Ejecutar TabNet sobre el espacio original y sobre el espacio post-DANN; comparar los `M_agg` globales. Si una feature pierde peso predictivo desproporcionado tras adaptar, es candidata a transferencia negativa. |
| ¿Las métricas Trustworthiness, Continuity, `Q_NX`, `R_NX` (§8.4.1) del pre-DANN vs. post-DANN cambian? | Reusar exactamente las métricas de §8.4.1 para cuantificar pérdida de estructura local tras la adaptación. |

**En síntesis para Fase 2:** la tesis no entrega herramienta de adaptación, pero entrega un **kit de diagnóstico geométrico** que el proyecto puede aplicar para validar empíricamente qué tan bien funciona DANN/CDAN sobre datos ICFES. La estrategia central de adaptación sigue siendo la del proyecto original.

---

### 2.3 Fase 3 — XAI / Interseccionalidad (OE3)

Aquí se concentran los aportes más visualmente potentes de la tesis. El proyecto contempla SHAP condicional + feature grouping; la tesis ofrece **dos complementos**:

#### 2.3.1 TabNet-informed Local Biplot como complemento a SHAP

| Aporte tesis (§8.3) | Aplicación ICFES |
|---|---|
| Sparsemax en M[i] → selección rala de features | Mitiga **directamente la multicolinealidad** del proyecto (limitación 1, contexto §8.1): TabNet elige un representante por grupo correlacionado en lugar de repartir importancia entre ellos como SHAP marginal. |
| Sobreposición de M_agg sobre flechas del biplot | Reemplaza el "diagrama de enjambre multifactorial" propuesto en §5.5 del proyecto con un **biplot interseccional**: clusters = perfiles territoriales-funcionales; flechas = features socioeconómicas; intensidad = atención predictiva. |
| Comparación con embeddings RF/LR/TabNet (§8.3) | Triangular interpretabilidad: SHAP global + TabNet M_agg + Local Biplot ψ_c — si los tres convergen, la inferencia es robusta. |

**Cita justificatoria (§8.5.4, p. 112):**
> *"By fusing TabNet's attention mechanisms with the topological clarity of the UMAP-based Local Biplot, the underlying logic of the model is laid bare."*

Para el proyecto ICFES, la "logic laid bare" se traduce en: *qué intersección de carencias pesa más en qué tipo de territorio*.

#### 2.3.2 Local Biplot puro como visualización interseccional

Aun sin TabNet, el **Local Biplot puro (§7)** ya entrega lo que el proyecto necesita para el OE3:
- Cada cluster `c` en el UMAP es un **perfil interseccional emergente**.
- El ranking `ψ_c = |B_c| · 1` (Ec. 7-8) es directamente una **matriz de vulnerabilidad interseccional** (§5.5 del contexto).
- La correlación local `ϱ̃_jj'` (Ec. 7-9 reaplicada sobre `B̃_c`) revela **qué pares de features interactúan dentro de cada perfil** — exactamente la operacionalización de "efecto multiplicativo de carencias cruzadas" que el proyecto necesita demostrar.

**Cita justificatoria (§7.4.1, p. 81):**
> *"Crucially, to assess how local embeddings capture nonlinear interactions, a nonlinear Local Biplot correlation ϱ̃_{jj'} is introduced. This metric is computed by substituting the original feature vectors in Eq. (7-9) with the rows of the aligned basis matrix B̃_c obtained from our method. By comparing ϱ (global) and ϱ̃ (local), the enhanced representational power of the proposed approach is demonstrated."*

---

## 3. Cómo cada aporte fortalece el marco conceptual del proyecto

| Concepto teórico ICFES | Refuerzo desde la tesis |
|---|---|
| **Territorios funcionales** (§5.1) — "se abandona la dicotomía urbano/rural estática" | El UMAP-based Local Biplot **operacionaliza geométricamente** territorios funcionales emergentes en lugar de imponerlos. Cita: *"localized phenological patterns must be preserved without collapsing the global variability of the dataset"* (§7.2, p. 80) — sustituir "phenological" por "territorial" es el calco exacto. |
| **Interseccionalidad** (§5.2) — "la desigualdad no es aditiva" | La correlación local no-lineal `ϱ̃_{jj'}` (Ec. 7-9 aplicada localmente) **cuantifica numéricamente** la interacción entre features por cluster. Esto es la traducción computacional directa de "efecto multiplicativo de carencias cruzadas". |
| **XAI / SHAP** (§5.5) | TabNet ofrece interpretabilidad nativa (no post-hoc como SHAP), con sparsemax (Ec. 8-2) que controla multicolinealidad por selección rala. El TabNet-informed Local Biplot combina visualización geométrica (Local Biplot) con relevancia supervisada (M_agg de TabNet). |
| **Equidad algorítmica / Fairness** (§5.4) | La tesis **no aporta** sobre fairness algorítmica en sentido estricto. La pieza más cercana —el reliability gating del M-GCECDL— se descarta por las razones expuestas en §1.3. El concepto de fairness del proyecto se mantiene con su marco original. |

---

## 4. Resolución de las tres limitaciones reconocidas del proyecto

El contexto del proyecto (§8) reconoce tres limitaciones. La tesis ofrece soluciones técnicas para las dos primeras:

### Limitación 1 — Multicolinealidad sociodemográfica
> *"En Colombia, estrato, falta de internet y región están fuertemente correlacionadas. SHAP marginal puede generar escenarios sintéticos socialmente imposibles."*

**Solución propuesta en el proyecto:** SHAP condicional + feature grouping.
**Refuerzo desde la tesis:**
- **TabNet sparsemax** (§8.1, Ec. 8-2) elige features ralamente, mitigando el reparto espurio de importancia entre features correlacionadas.
- **Local Biplot por cluster** (§7.3): la correlación se calcula dentro de cada cluster funcional, donde la multicolinealidad es estructuralmente menor que en el dataset global.

### Limitación 2 — Riesgo de transferencia negativa
> *"La DANN podría borrar características correlacionadas con el territorio pero legítimamente predictivas de competencias cognitivas."*

**Solución propuesta en el proyecto:** Adaptación de Dominio Condicional (CDAN).
**Refuerzo desde la tesis:** **la tesis no aporta sobre adaptación de dominio.** La pieza que en una primera versión de este documento se propuso aquí (M-GCECDL) se descarta por la justificación de §1.3 — el M-GCECDL no es una herramienta de adaptación de dominio, es de fusión multimodal con reliability gating, y los problemas son estructuralmente distintos. Lo que sí puede aportar la tesis es **diagnóstico empírico** de la transferencia negativa, comparando el `M_agg` de TabNet pre- y post-CDAN para identificar features cuya importancia predictiva se derrumba tras la adaptación (ver §2.2).

### Limitación 3 — Variables omitidas estructuralmente
> *"El dataset no captura salud mental, violencia intrafamiliar ni calidad pedagógica docente."*

**La tesis no resuelve esta limitación** porque es propia de la cobertura del dato, no del método. Sin embargo, la tesis sí ofrece una **estrategia metodológica honesta**: en §10.1 *Threats to Validity* (p. 152) Triana-Martinez modela explícitamente sus *threats* en tres categorías (internal, external, construct) — replicar esa estructura en el reporte ICFES fortalece la honestidad metodológica.

---

## 5. Pipeline técnico integrado propuesto para ICFES

Sintetizando el mapeo anterior, propongo el siguiente pipeline integrado para las 30 semanas del proyecto, que combina los métodos originales del proyecto con los aportes de la tesis:

| Sem. | Fase ICFES | Acción original del proyecto | Refuerzo desde la tesis |
|---|---|---|---|
| 1–4 | Preprocesamiento | Limpieza, normalización geográfica | — |
| 5–8 | F1: Domain Shift | XGBoost + MLP base | **+ UMAP global** sobre 28 features; medir Trustworthiness, Continuity, Q_NX(K), AUC[R_NX(K)] (Ecs. 8-19 y 8-20) como diagnóstico topológico del shift. |
| 9–14 | F2: DANN/CDAN | DANN clásico + CDAN como refinamiento | **Diagnóstico complementario** (no sustitutivo): extraer embeddings pre/post adaptación; aplicar Local Biplot al embedding final para visualizar si las flechas de `cole_area_ubicacion` se acortan tras adaptar (evidencia de invariancia). |
| 15–18 | F3: XAI | SHAP condicional + grouping | **+ TabNet entrenado por separado**; extraer `M_agg` (Ec. 8-12); correlacionar con SHAP global como triangulación de interpretabilidad. |
| 19–22 | F3: Visualización | Diagrama de enjambre | **+ TabNet-informed Local Biplot**: clusters = perfiles territoriales-funcionales emergentes; flechas = features; color = atención TabNet. Generar la "matriz de vulnerabilidad interseccional" desde `ψ_c` (Ec. 7-8). |
| 23–25 | Reporting | Working Paper, Policy Brief | Replicar estructura "Threats to Validity" (§10.1) del trabajo base. |

---

## 6. Limitaciones del mapeo (lo que NO se traslada directamente)

Por honestidad metodológica, conviene señalar diferencias entre los dominios que afectan la traducción aún de las dos contribuciones que sí se trasladan:

1. **Naturaleza del dato.** ICFES son 28 features tabulares de un único cuestionario; la tesis combina RGB + multiespectral + térmico (canales sensoriales heterogéneos). Esa diferencia es la razón principal para descartar el M-GCECDL del Cap. 9 (ver §1.3). Para Local Biplot (Cap. 7) y TabNet-informed Local Biplot (Cap. 8) la diferencia es menos crítica porque ambos son herramientas de visualización/atención agnósticas al origen físico de los datos.

2. **Regresión vs. clasificación — sin fricción para Caps. 7 y 8.** Los dos capítulos que sí se trasladan son **regresión nativa**: Cap. 7 valida el Local Biplot con RF/LR sobre breeding score y CWC (continuos); Cap. 8 define explícitamente `f_reg : R^P → R` con MSE como pérdida y R² ≈ 0.77–0.79 como métrica (§8.1 p. 93, Ec. 8-9). ICFES regresa `punt_global` (continuo), así que el paradigma de target coincide sin necesidad de adaptación.

3. **Escala de datos.** ICFES tiene 4.82M registros; la tesis trabaja sobre miles. UMAP escala mal con N grande; será necesario subsampling estratificado o usar **PaCMAP** / variante batch de UMAP. Esta limitación operacional debe declararse.

4. **Naturaleza del cluster.** En la tesis, los clusters corresponden a estadios fenológicos (T1–T6); en ICFES corresponderían a perfiles territoriales-socioeconómicos que **no tienen ground truth**. La validación interna se hará con silhouette y stability bootstrapping, no con etiqueta externa.

---

## 7. Conclusiones para la presentación

Cinco puntos que recomendaría llevar a la defensa frente al director:

1. **La tesis aporta dos piezas trasladables, no cinco.** Las contribuciones que sobreviven a un análisis honesto son el **UMAP-based Local Biplot (Cap. 7)** y el **TabNet-informed Local Biplot (Cap. 8)**. Ambas son herramientas de visualización e interpretabilidad agnósticas al dominio sensorial, por lo que su traslado a datos educativos no exige forzar analogías.

2. **El M-GCECDL del Cap. 9 se descarta por inconsistencia conceptual.** Está diseñado para fusionar canales sensoriales físicamente heterogéneos (RGB + multiespectral + térmico). ICFES tiene un único canal de adquisición (el formulario Saber 11), y su problema (sesgo urbano vs. rural) es covariate shift dentro del mismo canal — pertenece al dominio de DANN/CDAN, no de la fusión multimodal. Declarar esto en la defensa es una decisión metodológica fuerte, no una debilidad.

3. **TabNet-informed Local Biplot complementa SHAP**, no lo reemplaza. La triangulación SHAP condicional + `M_agg` de TabNet + `ψ_c` local da robustez metodológica defendible y mitiga la multicolinealidad por la selección rala del sparsemax.

4. **Operacionalización geométrica de "territorios funcionales".** El concepto teórico-político del proyecto encuentra en el UMAP + K-means latente un correlato técnico riguroso, no metafórico — los clusters territoriales-funcionales **emergen empíricamente** en lugar de imponerse por la etiqueta administrativa `cole_area_ubicacion`.

5. **Para Fase 2 (Adaptación de Dominio), la tesis aporta diagnóstico, no método.** El proyecto mantiene DANN y CDAN como herramientas centrales; lo que la tesis añade es un kit de verificación geométrica para auditar visualmente si la adaptación efectivamente reduce la separabilidad de dominio sin colapsar señal predictiva.

---

## 8. Elementos de la tesis que NO aplican al proyecto ICFES

Por honestidad metodológica frente al comité, conviene declarar explícitamente qué partes del trabajo base **no** son trasladables. Las agrupo en cuatro categorías según la razón de inaplicabilidad.

### 8.1 Inaplicable por naturaleza del dato (sensorial vs. tabular)

| Elemento de la tesis (dentro del alcance Caps. 7+) | Por qué no aplica |
|---|---|
| **Todos los índices de vegetación** usados como features en Caps. 7, 8 y 9 (NDVI, NDRE, MCARI2, PSRI, SAVI, NDLuv, RCC, GCC, BCC, ExG, VEG, CWSI, WI, MGRVI, CIVE, etc.) | Son combinaciones de bandas espectrales de plantas. No existen análogos en variables socioeconómicas. Las "features derivadas" en ICFES son lógicamente distintas (índices INSE/NSE), y el proyecto explícitamente las excluye (§6.3 del contexto). |
| **Multimodalidad sensorial (RGB + Multiespectral + Térmico + LiDAR)** del M-GCECDL (Cap. 9) | ICFES es **single-modal en el sentido sensorial**. La partición en cuatro "modalidades" `cole/estu/fami/temporal` para M-GCECDL es conceptual, no sensorial. Se justifica porque §9 (p. 114) admite particiones "*disjoint or partially overlapping*", pero la riqueza informativa entre modalidades será menor que entre sensores físicos distintos. |
| **Threats sobre "destructive sampling" y "mixed-pixel effects"** (§10.1, p. 152) | Son sesgos físicos de muestreo agrícola. En ICFES los sesgos son administrativos (autorreporte, no comparecencia al examen, errores de registro) — categorías completamente distintas. |
| **Future work sobre hiperespectral y LiDAR** (§8.6, p. 113) | Sin correlato en datos educativos. |

### 8.2 Inaplicable por naturaleza del problema (M-GCECDL completo)

El M-GCECDL del Cap. 9 **se descarta íntegro** por la justificación de §1.3 (no es una herramienta de adaptación de dominio; sus modalidades son canales sensoriales heterogéneos y los grupos de features de ICFES no lo son). En consecuencia, todos los componentes específicos de su arquitectura quedan fuera:

| Componente del M-GCECDL descartado | Razón |
|---|---|
| **Encoders modulares `E_m`, predictores `P_m`, reliability gates `R_m`** (Ecs. 9-1 a 9-3) | La arquitectura modular asume canales con principios físicos distintos. En ICFES no hay tal heterogeneidad. |
| **Fusión Product-of-Experts** (Ec. 9-4) | Ponderar predicciones por canal sólo tiene sentido si los canales son fuentes informativas independientes. |
| **Función objetivo compuesta** (Ec. 9-5): `L_GCE` primaria + supervisión modal + GJS Agreement + KL/H regularization + `L_corr` co-regularization | Cada término asume estructura modular sensorial. |
| **Generalized Jensen-Shannon divergence (GJS)** (Ec. 9-6) | Mide acuerdo entre distribuciones softmax de modalidades; no aplica. |
| **Hiperparámetro de temperatura `T`**, coeficientes `γ_sup, γ_agr, γ_reg, γ_co, τ, δ`, truncamientos `q, q_d` | Calibrados para el equilibrio de cinco objetivos modales que ICFES no necesita. |
| **Métricas de validación BACC, NMI, Cohen's kappa, Confusion Matrix** (usadas en Cap. 9) | Son métricas de clasificación. ICFES usa RMSE, MAE, R², stratified MAE por subgrupo. |

**Lo que sí sobrevive (Cap. 8): regresión nativa con TabNet.** El Cap. 8 ya está formulado como `f_reg : R^P → R` con MSE (Ec. 8-9) y R² ≈ 0.77–0.79 (§8.6 p. 112), así que para ICFES no hay fricción de paradigma. La única recalibración necesaria es ajustar el balance `λ_sparse` de `L_sparse` (Ec. 8-10) a la escala dinámica de `punt_global`.

### 8.3 Inaplicable por escala (miles vs. millones de registros)

| Elemento de la tesis | Por qué no aplica |
|---|---|
| **UMAP con `Spectral Embedding` como inicialización** (Algoritmo 1, p. 80) | El Spectral Embedding requiere descomposición de un Laplaciano `N×N`. Con N = 4.82M es computacionalmente prohibitivo. Hay que usar PCA initialization, batched UMAP, **PaCMAP** o **TriMap**. |
| **Hiperparámetros típicos de UMAP** `k ∈ {10, 15, 30}` neighbors | Calibrados para N ~ 10³–10⁴. Para N = 10⁶, valores tan bajos producen embeddings ruidosos. Habrá que escalar a `k ∈ {100, 200}` o usar subsampling estratificado urbano/rural. |
| **Diez corridas independientes con seeds distintos** (§8.4.1, p. 99) | Para evaluar estabilidad — viable en la tesis con datasets pequeños. Con 4.82M registros y arquitectura tipo M-GCECDL, diez entrenamientos completos son inviables en tiempo razonable. Habrá que reducir a 3–5 corridas o usar bootstrapping selectivo. |
| **Búsqueda de hiperparámetros TabNet por grid search exhaustivo** (`N_d = N_a ∈ {3,...,64}, γ ∈ {1,...,2}, B ∈ {16,...,1024}, N_steps ∈ {1,...,10}`, §8.4.1) | Es prohibitivo a escala ICFES. Habrá que usar búsqueda bayesiana (Optuna) o random search con presupuesto fijo. |
| **K-means determinístico clásico** (Ec. 7-6) | Para 4.82M puntos no escala bien sin Mini-Batch K-means o variante GPU. |

### 8.4 Inaplicable por sentido conceptual (no por tecnicidad)

| Elemento de la tesis | Por qué no aplica |
|---|---|
| **Marco completo de fusión multimodal del Cap. 9 (M-GCECDL)** | Ya argumentado en §1.3 y §8.2: está diseñado para canales sensoriales heterogéneos; ICFES tiene un único canal de adquisición (Saber 11). El problema que ICFES necesita resolver (urbano vs. rural) es covariate shift, no fusión multimodal. |
| **"Drought stress detection" y "phenological transition tracking"** como tareas semánticas | El proyecto ICFES no busca eventos transitorios. Predice un puntaje estructural producto de años de acumulación social. La "no-estacionariedad" en ICFES es **temporal lenta** (cambio de cohortes 2014→2024), no episódica. |
| **High-Throughput Phenotyping (HTP)** como objetivo aplicado (§9.3, p. 150) | Transformación operacional agronómica (drones reemplazando muestreo manual). El examen Saber 11 ya es censal y digital — no hay correlato operacional. |
| **"Validation in operational settings will require sustained collaboration with agricultural practitioners"** (§7.6, p. 92) | El correlato en ICFES sería diálogo con tomadores de decisión de política educativa, pero la **interpretación social del SHAP/Local Biplot ICFES** requiere consideraciones éticas (estigmatización por perfilamiento) que la tesis agrícola no enfrenta. La dinámica "validation with stakeholders" no es trasladable sin adaptación ética. |
| **Selección de cultivares por scoring** (§8.5.2, p. 111) | El "breeding score" es seleccionar individuos superiores para reproducción. **No hay análogo ético** en educación — usar predicciones individuales para "seleccionar" estudiantes sería discriminación algorítmica directa. El proyecto ICFES debe declarar explícitamente que las predicciones NO se usan para clasificar estudiantes, sino para diagnosticar territorios e infraestructura. |

### 8.5 Síntesis: ¿qué queda y qué se descarta?

Se **traslada** (Caps. 7 y 8):
- **UMAP-based Local Biplot** (Cap. 7) con escalamiento a 4.8M registros (subsampling estratificado o variante batch de UMAP)
- **TabNet-informed Local Biplot** (Cap. 8) — Cap. 8 ya es regresión nativa; sólo re-calibrar `λ_sparse` para la escala dinámica de `punt_global`
- **Métricas de evaluación geométrica** (Trustworthiness, Continuity, `Q_NX`, `R_NX`, AUC, GNN gain — Ecs. 8-19 y 8-20) como kit de diagnóstico para Fases 1 y 2
- **Estructura del Cap. 10 (Threats to Validity)** como modelo para el reporte final del proyecto

Se **descarta completamente**:
- **M-GCECDL del Cap. 9** y todos sus componentes (encoders modulares, reliability gates, fusión PoE, GJS, `L_GCE`, `L_corr`, hiperparámetros asociados) — justificado en §1.3
- Índices de vegetación específicos usados como features (NDVI, NDRE, MCARI2, etc.)
- Marco aplicado de selección de cultivares (§8.5.2) por incompatibilidad ética en educación
- Inicializaciones y rangos de hiperparámetros de UMAP y TabNet incompatibles con la escala ICFES
- Threats agronómicos del §10.1 (destructive sampling, mixed-pixel effects)

---

## 9. Referencias clave de la tesis usadas en este documento

- Triana-Martinez, J. C. (2026). *A deep learning approach with preserved interpretability to support multimodal remote sensing in precision agriculture with applications to the Tolima region.* PhD Thesis. **Capítulos considerados como aplicables: 7, 8 y 10.** (El Cap. 9 se examina sólo para justificar su descarte en §1.3.)
- Triana-Martinez et al. (2024). *Crop Water Status Analysis from Complex Agricultural Data using UMAP-based Local Biplot.* (base del Cap. 7).
- Triana-Martinez et al. (2025). *TabNet-informed Local Biplot for explainable crop trait estimation.* (base del Cap. 8).
- McInnes, L. et al. (2018). UMAP (§7.2).
- Arik, S. & Pfister, T. (2021). TabNet (§8.1).
- Martins & Astudillo (2016). Sparsemax (§8.1, Ec. 8-2).
- Lee & Verleysen (2009); Lee et al. (2015); De Bodt et al. (2019). Métricas Q_NX, R_NX, GNN gain (§8.4.1).
