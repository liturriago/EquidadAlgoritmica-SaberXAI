"""
Script para entrenar modelos separados por clúster LCA (territorios funcionales).
Itera sobre las 7 clases LCA (0-6) y entrena XGBoost + MLP para cada una.

Wrapper legacy de ``scripts/train.py --mode cluster``: conserva la CLI
original para no romper notebooks ni el handoff del entregable 3.
"""
import argparse
from pathlib import Path

from saber_xai.config import config

from train import run_cluster_training


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento de modelos por clúster LCA"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lca_clusters.yaml",
        help="Ruta al archivo de configuración YAML"
    )
    parser.add_argument(
        "--clusters",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5, 6],
        help="IDs de clústeres LCA a entrenar (default: todos 0-6)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directorio donde se guardarán los modelos"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directorio donde se guardarán métricas y metadatos"
    )
    parser.add_argument(
        "--allow-dummy",
        action="store_true",
        help="Permitir datos sintéticos si falla la carga (solo pruebas)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Cargando configuración desde: {args.config}")
    config.load_from_yaml(args.config)
    if args.allow_dummy:
        config.allow_dummy_data = True

    models_dir = Path(args.models_dir)
    results_dir = Path(args.results_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Entrenando modelos para clústeres: {args.clusters}")
    print(f"Modelos se guardarán en: {models_dir}/")

    run_cluster_training(args.clusters, models_dir, results_dir)

    print(f"\n{'='*60}")
    print("  ENTRENAMIENTO COMPLETADO")
    print(f"{'='*60}")
    print(f"Modelos guardados en: {models_dir}/")
    for cluster_id in args.clusters:
        print(f"  Clúster {cluster_id}: xgb_cluster_{cluster_id}.json, mlp_cluster_{cluster_id}.pth")


if __name__ == "__main__":
    main()
