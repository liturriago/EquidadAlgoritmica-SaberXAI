"""
Script orquestador principal.
"""
import argparse
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
    print("\n--- Fase 1: Ingesta de Datos y Target Encoding ---")
    data_module = DataModule()
    data_module.prepare_data()
    
    train_loader, val_loader, test_loader = data_module.get_dataloaders()
    dtrain, dval, dtest = data_module.get_dmatrices()
    
    print(f"Datos de entrenamiento: {data_module.X_train.shape[0]} muestras, {data_module.X_train.shape[1]} features.")
    
    # 2. XGBoost
    print("\n--- Fase 2: Entrenamiento XGBoost ---")
    xgb_model = XGBoostModel()
    xgb_model.train_cv(dtrain)
    xgb_model.save_model()
    
    # 3. PyTorch MLP
    print("\n--- Fase 3: Entrenamiento MLP en PyTorch ---")
    input_dim = data_module.X_train.shape[1]
    mlp = MLP(input_dim=input_dim)
    
    trainer = MLPTrainer(model=mlp)
    trainer.train(train_loader=train_loader, val_loader=val_loader)
    
    # 4. Evaluación
    print("\n--- Fase 4: Evaluación de Modelos (Test Set) ---")
    evaluator = ModelEvaluator()
    evaluator.evaluate_xgb(xgb_model.model, dtest)
    
    # Cargar los mejores pesos para MLP si se guardaron en el early stopping
    try:
        mlp.load_state_dict(torch.load("best_mlp_model.pth", weights_only=True))
    except FileNotFoundError:
        pass # Si no hay, evalúa con los pesos de la última época
        
    evaluator.evaluate_mlp(mlp, test_loader)
    
    print("\nPipeline completado exitosamente.")

if __name__ == "__main__":
    main()
