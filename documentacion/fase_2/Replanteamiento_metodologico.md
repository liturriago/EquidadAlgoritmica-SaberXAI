# Replanteamiento metodológico del proyecto

### De DANN a un enfoque de territorios funcionales, robustez por grupo e interpretabilidad

*Proyecto de equidad algorítmica sobre Saber 11 — 21 de mayo de 2026*

---

## 1. La idea original

El problema que nos propusimos resolver fue **explicar y dar interpretabilidad a la diferencia de puntajes entre estudiantes urbanos y rurales** en Saber 11. Para ello planteamos una arquitectura basada en dos herramientas: una red adversarial de dominio (DANN) y valores SHAP.

Esa arquitectura se sostenía sobre una hipótesis: que la diferencia urbano/rural era un problema de *domain shift*: lo rural sería un dominio subrepresentado y más difícil, y la menor cantidad de datos haría que los modelos predictivos rindieran peor allí — es decir, que la cantidad de datos "colapsaría" el sistema en lo rural. DANN se encargaría de adaptar el modelo entre dominios, y SHAP aportaría la explicabilidad.

## 2. Por qué cambiamos de decisión

Dos hallazgos empíricos de nuestro propio análisis le retiran el sustento a ese enfoque.

### 2.1 UMAP: lo urbano y lo rural no son diferenciables

Una proyección UMAP sobre las 26 covariables, coloreada por área, muestra que los puntos urbanos y rurales se solapan fuertemente: no forman regiones separables en el espacio de covariables. Esto golpea directamente el supuesto de DANN, cuyo mecanismo depende de que los dominios sean distinguibles — su discriminador de dominio necesita poder separarlos para que la señal adversarial signifique algo. Si lo urbano y lo rural no son separables, esa señal es débil y DANN tiene poco que hacer.

### 2.2 MLP y XGBoost rinden bien en lo rural

Al entrenar una MLP y un modelo XGBoost, el rendimiento sobre el subconjunto rural no se degradó: fue comparable o mejor que en lo urbano. Esto desmiente la hipótesis de que la cantidad o el desbalance de los datos colapsaría los modelos en lo rural. No existe el "colapso de dominio" que DANN debería reparar.

### 2.3 Conclusión

Los dos pilares que justificaban a DANN — un *domain shift* separable y un rendimiento degradado en el grupo desfavorecido — fallan empíricamente. A esto se suma una razón conceptual: DANN es una herramienta diseñada para *eliminar* la información de dominio de la representación, lo cual es lo opuesto a nuestro objetivo real, que es hacer la desigualdad **visible y explicable**.

El hallazgo de fondo es otro, y es más interesante: **las etiquetas administrativas (urbano/rural) son ciegas a la verdadera topología de la desigualdad.** La vulnerabilidad no se alinea con la frontera burocrática urbano/rural; la trasciende. Sobre esa idea reorientamos la propuesta.

## 3. La nueva propuesta

El objetivo se mantiene — explicar la brecha de puntajes y aportar interpretabilidad —, pero ahora se construye sobre la estructura real de la desigualdad y no sobre una etiqueta administrativa. La propuesta tiene tres componentes encadenados.

### 3.1 Modelo no supervisado: descubrir los territorios funcionales

Aplicamos *clustering* sobre las covariables socioeconómicas y familiares para descubrir **territorios funcionales latentes**, en lugar de imponer el binario urbano/rural. Estos clústeres se convierten en los subgrupos significativos del análisis — por ejemplo, un territorio de alta vulnerabilidad estructural y uno de privilegio — y pueden cruzar la línea urbano/rural. Este paso es la columna vertebral empírica de la tesis: es donde se establece y se valida el argumento central.

Como las covariables son mayormente categóricas y ordinales, los métodos candidatos incluyen el Análisis de Clases Latentes (LCA) o el *clustering* con distancia de Gower; conviene verificar la estabilidad de la solución y, de ser posible, validarla contra índices externos como el SISBEN o el IPM.

### 3.2 Red neuronal (MLP) con función de pérdida Group DRO

Entrenamos una red neuronal — una MLP — cuya función de pérdida es el objetivo **Group DRO** (*Distributionally Robust Optimization*). El entrenamiento estándar minimiza el error promedio y tiende a favorecer a los grupos mayoritarios; Group DRO, en cambio, optimiza el error del grupo que peor va, de modo que el modelo también prediga bien en los territorios funcionales más vulnerables.

La etiqueta de clúster se usa dentro del objetivo DRO como índice de grupo — para calcular el error por grupo e identificar el peor —, **no como variable de entrada**. La red predice `punt_global` únicamente a partir de las covariables.

Esto es clave para la interpretabilidad: un buen rendimiento por grupo es la precondición para que las explicaciones sean confiables. SHAP explica fielmente al modelo; si el modelo predijera mal a un grupo, su explicación para ese grupo estaría explicando una predicción equivocada. DRO garantiza que el modelo sirva a cada grupo, de manera que las explicaciones sean válidas en todos — y no sesgadas hacia la mayoría.

### 3.3 SHAP segmentado por territorios

Calculamos los valores SHAP y los agrupamos por territorio funcional, para identificar **qué variables son más relevantes dentro de cada subgrupo**. Esto revela la importancia asimétrica de las variables: una misma variable puede pesar de forma muy distinta según el territorio. El entregable para el ICFES es concreto: en cada territorio funcional, cuáles son las variables donde se concentra la brecha y, por lo tanto, dónde una intervención tendría mayor palanca.

## 4. Consideraciones de implementación

El orden no es arbitrario: forma una cadena de dependencias. El modelo no supervisado *define* los grupos; la MLP con Group DRO *consume* esos grupos para predecir bien en cada uno; el SHAP segmentado *consume* esos mismos grupos para interpretar. Ningún paso posterior puede ejecutarse sin el anterior.

En términos de datos, el conjunto de entrenamiento tiene tres tipos de columnas: las covariables (entran a la red como input), la etiqueta `punt_global` (lo que la red aprende a predecir) y el clúster (metadato que usan el objetivo DRO y la segmentación de SHAP, pero que la red nunca recibe como entrada).

El rendimiento del modelo debe validarse **por clúster**, no solo de forma global; una eventual disparidad de rendimiento entre territorios es, en sí misma, un hallazgo coherente con el enfoque de equidad. Conviene además recordar que DRO suele intercambiar algo de precisión promedio por mejor rendimiento del peor grupo: es una decisión deliberada y honesta, alineada con el objetivo de equidad.

Por último, SHAP es una herramienta **descriptiva, no causal**: indica dónde se concentra la brecha y cuáles son las variables de mayor palanca según las asociaciones que aprendió el modelo. La recomendación al ICFES debe presentarse como un conjunto de variables prioritarias y, en lo posible, complementarse con un razonamiento causal explícito.

## 5. Nota: DANN como valor agregado

DANN no se descarta como concepto. Tiene un uso legítimo que no encaja con el objetivo explicativo de esta tesis, pero que podría incorporarse como un módulo complementario. Si el ICFES quisiera determinar el rendimiento de estudiantes **de manera individual** en un escenario operativo — por ejemplo, desplegar un modelo predictivo sobre una nueva cohorte o un contexto del que aún no se tienen resultados (etiquetas) —, la capacidad de adaptación de dominio de DANN podría ayudar a que ese predictor generalice entre contextos. Sería un componente de valor agregado para la predicción a nivel individual en producción, separado del núcleo de interpretabilidad de este trabajo.

---

*Referencia para Group DRO: Sagawa et al. (2020), "Distributionally Robust Neural Networks for Group Shifts".*
