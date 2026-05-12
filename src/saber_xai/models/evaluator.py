"""
Módulo para la evaluación del rendimiento de los modelos.
"""
import torch
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader
import torch.nn as nn

class ModelEvaluator:
    """Clase utilitaria para evaluar modelos estandarizadamente."""
    
    def __init__(self, device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate_xgb(self, model: xgb.Booster, dtest: xgb.DMatrix, test_groups: np.ndarray = None):
        """Evalúa el modelo XGBoost en el conjunto de prueba y calcula métricas."""
        print("Evaluando XGBoost en test_loader...")
        y_pred = model.predict(dtest)
        y_true = dtest.get_label()
        
        self._evaluate_groups(y_true, y_pred, "XGBoost", test_groups)

    def evaluate_mlp(self, model: nn.Module, test_loader: DataLoader, test_groups: np.ndarray = None):
        """Evalúa el modelo PyTorch MLP en el conjunto de prueba."""
        print(f"Evaluando PyTorch MLP en {self.device}...")
        model.to(self.device)
        model.eval()
        
        y_true_list = []
        y_pred_list = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = model(X_batch)
                
                y_pred_list.append(outputs.cpu().numpy())
                y_true_list.append(y_batch.cpu().numpy())
                
        # Unir todos los batches
        y_true = np.vstack(y_true_list)
        y_pred = np.vstack(y_pred_list)
        
        self._evaluate_groups(y_true, y_pred, "PyTorch MLP", test_groups)

    def _evaluate_groups(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, test_groups: np.ndarray):
        """Evalúa globalmente y por subgrupos si están disponibles."""
        self._print_metrics(y_true, y_pred, f"{model_name} (Global)")
        
        if test_groups is not None:
            # Es posible que y_true sea de dimensión [N, 1], mientras que mask es de dimensión [N]
            y_true_flat = y_true.flatten()
            y_pred_flat = y_pred.flatten()
            
            mask_rural = (test_groups == 'Rural')
            if np.any(mask_rural):
                self._print_metrics(y_true_flat[mask_rural], y_pred_flat[mask_rural], f"{model_name} (Rural)")
                
            mask_urbano = (test_groups == 'Urbano')
            if np.any(mask_urbano):
                self._print_metrics(y_true_flat[mask_urbano], y_pred_flat[mask_urbano], f"{model_name} (Urbano)")

    def _print_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
        """Calcula e imprime métricas de regresión incluyendo sesgo medio.
        
        El Mean Bias Error (MBE) es la métrica clave de equidad: un valor
        negativo indica que el modelo subestima sistemáticamente a ese grupo;
        positivo indica sobreestimación. El RMSE por sí solo no detecta esto.
        """
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        
        rmse = np.sqrt(mean_squared_error(y_true_f, y_pred_f))
        mae  = mean_absolute_error(y_true_f, y_pred_f)
        r2   = r2_score(y_true_f, y_pred_f)
        mbe  = np.mean(y_pred_f - y_true_f)  # + sobreestima, - subestima
        
        print(f"\n--- Resultados en Test: {model_name} ---")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R²:   {r2:.4f}")
        print(f"MBE:  {mbe:+.4f}  {'(⚠ sobreestima)' if mbe > 1 else '(⚠ subestima)' if mbe < -1 else '(sin sesgo relevante)'}")
        print("-" * 40)
