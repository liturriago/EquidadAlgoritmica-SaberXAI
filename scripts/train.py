"""
Script orquestador principal.
"""
import argparse
import gc
from saber_xai.config import config
from saber_xai.data.data_module import DataModule
from saber_xai.models.xgb_model import XGBoostModel
from saber_xai.models.mlp_model import MLP, MLPTrainer
from saber_xai.models.evaluator import ModelEvaluator
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline de ML para ICFES")
    parser.add_argument(
        "--config", 
        type=str, 
        help="Ruta al archivo de configuración YAML (ej. configs/default.yaml)", 
        default=None
    )
    return parser.parse_args()

def main():
    args = parse_args()
    if args.config:
        print(f"Cargando configuración desde: {args.config}")
        config.load_from_yaml(args.config)
        
    print("Iniciando Pipeline de ML para ICFES...")
    
    # 1. Preparación de datos
    print("\n--- Fase 1: Ingesta de Datos y Preprocesamiento ---")
    data_module = DataModule()
    data_module.prepare_data()
    
    print(f"Datos de entrenamiento: {data_module.X_train.shape[0]} muestras, {data_module.X_train.shape[1]} features.")
    
    # 2. XGBoost — Se crea primero y se libera antes de crear los DataLoaders de PyTorch.
    # Esto evita que DMatrices (copia numpy de XGBoost) y tensores de PyTorch
    # coexistan en RAM al mismo tiempo.
    print("\n--- Fase 2: Entrenamiento XGBoost ---")
    dtrain, dval, dtest = data_module.get_dmatrices()
    
    xgb_model = XGBoostModel()
    xgb_model.train_with_early_stopping(dtrain, dval)
    xgb_model.save_model()

    # Evaluación XGBoost — en este punto todavía no existen los tensores de PyTorch
    print("\n--- Fase 3: Evaluación XGBoost (Test Set) ---")
    evaluator = ModelEvaluator()
    evaluator.evaluate_xgb(xgb_model.model, dtest, test_groups=data_module.test_area)

    # Liberar DMatrices de XGBoost antes de crear los tensores de PyTorch
    del dtrain, dval, dtest
    gc.collect()
    
    # 3. PyTorch MLP — Ahora que XGBoost está listo y sus DMatrices liberados,
    # creamos los DataLoaders con los arrays numpy ya procesados.
    print("\n--- Fase 4: Entrenamiento MLP en PyTorch ---")
    input_dim = data_module.X_train.shape[1]
    train_loader, val_loader, test_loader = data_module.get_dataloaders()

    mlp = MLP(input_dim=input_dim)
    trainer = MLPTrainer(model=mlp)
    trainer.train(train_loader=train_loader, val_loader=val_loader)
    
    # 4. Evaluación MLP
    print("\n--- Fase 5: Evaluación MLP (Test Set) ---")
    # Cargar los mejores pesos si se guardaron en el early stopping
    try:
        mlp.load_state_dict(torch.load("best_mlp_model.pth", weights_only=True))
    except FileNotFoundError:
        pass  # Si no hay, evalúa con los pesos de la última época
        
    evaluator.evaluate_mlp(mlp, test_loader, test_groups=data_module.test_area)
    
    print("\nPipeline completado exitosamente.")

if __name__ == "__main__":
    main()
