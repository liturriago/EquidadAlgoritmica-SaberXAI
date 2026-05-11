# Planificador de Agentes - Proyecto Equidad Algorítmica Saber XAI

**Agente Supervisor**: Hola equipo. A continuación detallo el plan general y las tareas delegadas para la construcción del sistema predictivo del `punt_global` de las pruebas ICFES.

## Arquitectura y Estructura de Directorios

```plaintext
EquidadAlgoritmica-SaberXAI/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_module.py       # Carga de Parquet y Target Encoding (previniendo Data Leakage)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── xgb_model.py         # Pipeline de XGBoost con Cross-Validation
│   │   └── mlp_model.py         # Modelo Multi-Layer Perceptron en PyTorch con Early Stopping
│   ├── utils/
│   │   ├── __init__.py
│   │   └── config.py            # Manejo de configuración usando Pydantic
│   └── train.py                 # Script orquestador para ejecutar el entrenamiento
├── tests/
│   ├── __init__.py
│   ├── test_data.py             # Pruebas para evitar data leakage en Target Encoding
│   └── test_models.py           # Pruebas unitarias de inicialización de modelos
├── .github/
│   └── workflows/
│       └── ci.yml               # Configuración de GitHub Actions para pytest
├── pyproject.toml               # Dependencias e información del proyecto
└── planificador.md              # Bitácora/Reporte principal solicitado
```

## Estado de las Tareas Delegadas

### 1. Data Engineer
- [x] Implementar configuración centralizada con `Pydantic` en `src/utils/config.py`.
- [x] Implementar carga desde `icfes_final_serializado.parquet`.
- [x] Implementar Target Encoding en `cole_mcpio_ubicacion` ajustado exclusivamente sobre el conjunto de entrenamiento.
- [x] Exponer funciones para generar `DataLoader` (PyTorch) y `DMatrix` (XGBoost).

### 2. ML Researcher
- [x] Implementar entrenamiento de XGBoost con validación cruzada y guardado de modelo.
- [x] Implementar arquitectura y entrenamiento de MLP en PyTorch, configurable y con Early Stopping.
- [x] Escribir el orquestador principal `src/train.py`.

### 3. QA / DevOps
- [x] Crear el entorno del proyecto y definir `pyproject.toml` (PyTorch, XGBoost, etc.).
- [x] Implementar las pruebas unitarias usando `pytest` (verificando dataloaders, forward pass y Target Encoding).
- [x] Configurar el flujo de CI en `.github/workflows/ci.yml`.

> **Nota del Supervisor**: Todos los módulos deben tener comentarios en español técnico como se solicitó para su posterior revisión. Estaré documentando en este mismo archivo el progreso a medida que el equipo vaya completando las tareas.
