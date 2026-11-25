"""
Script orquestador de entrenamiento (XGBoost + MLP).

Modos:
  auto               resuelve según la config (global si hay data_dir, si no file)
  global             modelo único sobre la unión de los clústeres LCA
                     (mismos conjuntos de test que los modelos por clúster)
  global_indicators  igual que global + indicadoras one-hot del clúster LCA
  cluster            un modelo por clúster LCA (nombres xgb_cluster_{id}.json /
                     mlp_cluster_{id}.pth, compatibles con shap_analysis.py)
  file               modelo único sobre config.data_path (legacy)

Ejemplos:
  python scripts/train.py --config configs/global_baseline.yaml --mode global
  python scripts/train.py --config configs/global_baseline.yaml --mode global_indicators
  python scripts/train.py --config configs/lca_clusters.yaml --mode cluster

Salidas (por tag de corrida):
  models/xgb_{tag}.json(+.meta.json)   modelo XGBoost + features/best_iteration
  models/mlp_{tag}.pth                 mejores pesos del MLP (early stopping)
  results/metrics_{tag}.json           métricas global / área / clúster
  results/tabla4.csv                   filas acumuladas estilo Tabla 4
  results/split_meta_{tag}.json        tamaños y medias del split (trazabilidad)
  results/mlp_history_{tag}.json       curvas de entrenamiento del MLP
"""
import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from saber_xai.config import config
from saber_xai.data.data_module import DataModule
from saber_xai.models.xgb_model import XGBoostModel
from saber_xai.models.mlp_model import MLP, MLPTrainer
from saber_xai.models.evaluator import ModelEvaluator

MODES = ("auto", "global", "global_indicators", "cluster", "file")


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline de ML para ICFES")
    parser.add_argument(
        "--config",
        type=str,
        help="Ruta al archivo de configuración YAML (ej. configs/global_baseline.yaml)",
        default=None,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=MODES,
        default="auto",
        help="Modo de entrenamiento (default: auto, según la configuración)",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5, 6],
        help="IDs de clústeres LCA a entrenar en modo cluster (default: todos)",
    )
    parser.add_argument("--models-dir", type=str, default="models",
                        help="Directorio donde se guardarán los modelos")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directorio donde se guardarán métricas y metadatos")
    parser.add_argument("--tag", type=str, default=None,
                        help="Etiqueta de la corrida (default: derivada del modo)")
    parser.add_argument("--allow-dummy", action="store_true",
                        help="Permitir datos sintéticos si falla la carga (solo pruebas)")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Fija las semillas para reproducibilidad (numpy + torch)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single_training(tag: str, models_dir: Path, results_dir: Path) -> dict:
    """Entrena y evalúa XGBoost + MLP sobre el DataModule ya configurado.

    Usa la evaluación global, por área (Rural/Urbano) y por territorio funcional
    (clúster LCA) cuando ``dm.test_cluster_labels`` está disponible.
    """
    # 1. Preparación de datos
    print("\n--- Fase 1: Ingesta de Datos y Preprocesamiento ---")
    data_module = DataModule()
    data_module.prepare_data()

    n_samples = data_module.X_train.shape[0]
    n_features = data_module.X_train.shape[1]
    print(f"Datos de entrenamiento: {n_samples} muestras, {n_features} features.")
    if n_samples == 0:
        raise RuntimeError(f"Conjunto de entrenamiento vacío para el tag '{tag}'.")

    data_module.save_split_meta(str(results_dir / f"split_meta_{tag}.json"))
    evaluator = ModelEvaluator()
    reports = {}

    # 2. XGBoost — se entrena y evalúa antes de crear los tensores de PyTorch
    # para que DMatrices y tensores no coexistan en RAM.
    print("\n--- Fase 2: Entrenamiento XGBoost ---")
    dtrain, dval, dtest = data_module.get_dmatrices()

    xgb_model = XGBoostModel()
    xgb_model.train_with_early_stopping(dtrain, dval)

    xgb_path = models_dir / f"xgb_{tag}.json"
    xgb_model.save_model(str(xgb_path), feature_names=list(data_module.X_train.columns))

    print("\n--- Fase 3: Evaluación XGBoost (Test Set) ---")
    reports["xgboost"] = evaluator.evaluate_xgb(
        xgb_model, dtest,
        test_groups=data_module.test_area,
        cluster_labels=data_module.test_cluster_labels,
        model_name=f"XGBoost ({tag})",
    )

    del dtrain, dval, dtest
    gc.collect()

    # 3. PyTorch MLP
    print("\n--- Fase 4: Entrenamiento MLP en PyTorch ---")
    input_dim = data_module.X_train.shape[1]
    train_loader, val_loader, test_loader = data_module.get_dataloaders()

    mlp = MLP(input_dim=input_dim)
    trainer = MLPTrainer(model=mlp)
    checkpoint_path = models_dir / f"mlp_{tag}.pth"
    history = trainer.train(train_loader=train_loader, val_loader=val_loader,
                            checkpoint_path=str(checkpoint_path))
    with open(results_dir / f"mlp_history_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Modelo MLP guardado (mejores pesos): {checkpoint_path}")

    print("\n--- Fase 5: Evaluación MLP (Test Set) ---")
    reports["mlp"] = evaluator.evaluate_mlp(
        mlp, test_loader,
        test_groups=data_module.test_area,
        cluster_labels=data_module.test_cluster_labels,
        model_name=f"PyTorch MLP ({tag})",
    )

    # Liberar tensores antes de salir
    del mlp, trainer, train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Persistir métricas
    _save_metrics(reports, tag, data_module, results_dir)
    return reports


def run_cluster_training(clusters: list, models_dir: Path, results_dir: Path) -> None:
    """Entrena un modelo por clúster LCA (equivalente a --mode cluster)."""
    for cluster_id in clusters:
        print(f"\n{'=' * 60}")
        print(f"  CLÚSTER LCA {cluster_id}")
        print(f"{'=' * 60}\n")
        config.cluster_id = cluster_id
        run_single_training(tag=f"cluster_{cluster_id}", models_dir=models_dir,
                            results_dir=results_dir)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _save_metrics(reports: dict, tag: str, data_module: DataModule, results_dir: Path) -> None:
    """Guarda el JSON de métricas y añade filas al CSV tipo Tabla 4."""
    payload = {
        "tag": tag,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "split_meta": data_module.split_meta,
        "metrics": reports,
    }
    metrics_path = results_dir / f"metrics_{tag}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Métricas guardadas en {metrics_path}")

    tabla4_path = results_dir / "tabla4.csv"
    for model_key, report in reports.items():
        ModelEvaluator.append_tabla4(report, tag=tag, path=str(tabla4_path))


def main():
    args = parse_args()
    if args.config:
        print(f"Cargando configuración desde: {args.config}")
        config.load_from_yaml(args.config)
    if args.allow_dummy:
        config.allow_dummy_data = True

    set_seed(config.random_state)

    models_dir = Path(args.models_dir)
    results_dir = Path(args.results_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    mode = args.mode
    if mode == "auto":
        mode = "global" if config.data_dir is not None else "file"
        print(f"Modo auto resuelto: {mode}")

    print("Iniciando Pipeline de ML para ICFES...")
    print(f"Modo: {mode} | Modelos: {models_dir}/ | Resultados: {results_dir}/")

    if mode == "cluster":
        run_cluster_training(args.clusters, models_dir, results_dir)
    else:
        if mode in ("global", "global_indicators"):
            config.cluster_id = None
            config.territory_indicators = (mode == "global_indicators")
        elif mode == "file":
            config.cluster_id = None
            config.data_dir = None
        tag = args.tag or mode
        run_single_training(tag=tag, models_dir=models_dir, results_dir=results_dir)

    print("\nPipeline completado exitosamente.")
    print(f"Modelos en: {models_dir}/ | Métricas en: {results_dir}/")


if __name__ == "__main__":
    main()
