"""
Script para entrenar modelos separados por clúster LCA (territorios funcionales).
Itera sobre las 7 clases LCA (0-6) y entrena XGBoost + MLP para cada una.
"""
import argparse
import gc
import torch
from pathlib import Path

from saber_xai.config import config
from saber_xai.data.data_module import DataModule
from saber_xai.models.xgb_model import XGBoostModel
from saber_xai.models.mlp_model import MLP, MLPTrainer
from saber_xai.models.evaluator import ModelEvaluator


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
    return parser.parse_args()


def train_cluster(cluster_id: int, models_dir: Path) -> None:
    """Entrena modelos XGBoost y MLP para un clúster LCA específico."""
    print(f"\n{'='*60}")
    print(f"  CLÚSTER LCA {cluster_id}")
    print(f"{'='*60}\n")
    
    # Configurar cluster_id en la configuración global
    config.cluster_id = cluster_id
    
    # 1. Preparación de datos
    print(f"--- Fase 1: Datos del clúster {cluster_id} ---")
    data_module = DataModule()
    data_module.prepare_data()
    
    n_samples = data_module.X_train.shape[0]
    n_features = data_module.X_train.shape[1]
    print(f"Train: {n_samples:,} muestras, {n_features} features")
    
    if n_samples == 0:
        print(f"Advertencia: Clúster {cluster_id} vacío. Saltando...")
        return
    
    # 2. XGBoost
    print(f"\n--- Fase 2: XGBoost (clúster {cluster_id}) ---")
    dtrain, dval, dtest = data_module.get_dmatrices()
    
    xgb_model = XGBoostModel()
    xgb_model.train_with_early_stopping(dtrain, dval)
    
    # Guardar modelo con sufijo de clúster
    xgb_path = models_dir / f"xgb_cluster_{cluster_id}.json"
    xgb_model.model.save_model(str(xgb_path))
    print(f"Modelo XGBoost guardado: {xgb_path}")
    
    # Evaluación XGBoost
    print(f"\n--- Fase 3: Evaluación XGBoost (clúster {cluster_id}) ---")
    evaluator = ModelEvaluator()
    evaluator.evaluate_xgb(xgb_model.model, dtest, test_groups=data_module.test_area)
    
    # Liberar DMatrices
    del dtrain, dval, dtest, xgb_model
    gc.collect()
    
    # 3. PyTorch MLP
    print(f"\n--- Fase 4: MLP (clúster {cluster_id}) ---")
    input_dim = data_module.X_train.shape[1]
    train_loader, val_loader, test_loader = data_module.get_dataloaders()
    
    mlp = MLP(input_dim=input_dim)
    trainer = MLPTrainer(model=mlp)
    trainer.train(train_loader=train_loader, val_loader=val_loader)
    
    # Guardar modelo MLP con sufijo de clúster
    mlp_path = models_dir / f"mlp_cluster_{cluster_id}.pth"
    torch.save(mlp.state_dict(), mlp_path)
    print(f"Modelo MLP guardado: {mlp_path}")
    
    # 4. Evaluación MLP
    print(f"\n--- Fase 5: Evaluación MLP (clúster {cluster_id}) ---")
    try:
        mlp.load_state_dict(torch.load(str(mlp_path), weights_only=True))
    except FileNotFoundError:
        pass
    
    evaluator.evaluate_mlp(mlp, test_loader, test_groups=data_module.test_area)
    
    # Liberar tensores de PyTorch
    del mlp, trainer, train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"\nClúster {cluster_id} completado.")


def main():
    args = parse_args()
    
    # Cargar configuración
    print(f"Cargando configuración desde: {args.config}")
    config.load_from_yaml(args.config)
    
    # Crear directorio de modelos
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Entrenando modelos para clústeres: {args.clusters}")
    print(f"Modelos se guardarán en: {models_dir}/")
    
    # Entrenar cada clúster
    for cluster_id in args.clusters:
        train_cluster(cluster_id, models_dir)
    
    print(f"\n{'='*60}")
    print("  ENTRENAMIENTO COMPLETADO")
    print(f"{'='*60}")
    print(f"Modelos guardados en: {models_dir}/")
    for cluster_id in args.clusters:
        print(f"  Clúster {cluster_id}: xgb_cluster_{cluster_id}.json, mlp_cluster_{cluster_id}.pth")


if __name__ == "__main__":
    main()
