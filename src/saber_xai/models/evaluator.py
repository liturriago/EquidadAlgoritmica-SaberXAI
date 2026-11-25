"""
Módulo para la evaluación del rendimiento de los modelos.
"""
import json
import csv
import torch
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader
import torch.nn as nn

class ModelEvaluator:
    """Clase utilitaria para evaluar modelos estandarizadamente.

    Cada evaluación retorna un reporte estructurado:
    ``{"model", "global", "por_area", "por_cluster"}`` con métricas
    RMSE / MAE / R² / MBE y número de muestras. El MBE es la métrica clave
    de equidad: negativo = subestimación sistemática del grupo.
    """

    def __init__(self, device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate_xgb(self, model: xgb.Booster, dtest: xgb.DMatrix,
                     test_groups: np.ndarray = None, cluster_labels: np.ndarray = None,
                     model_name: str = "XGBoost") -> dict:
        """Evalúa el modelo XGBoost en el conjunto de prueba y calcula métricas."""
        print("Evaluando XGBoost en test_loader...")
        y_pred = model.predict(dtest)
        y_true = dtest.get_label()

        return self._build_report(y_true, y_pred, model_name, test_groups, cluster_labels)

    def evaluate_mlp(self, model: nn.Module, test_loader: DataLoader,
                     test_groups: np.ndarray = None, cluster_labels: np.ndarray = None,
                     model_name: str = "PyTorch MLP") -> dict:
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

        return self._build_report(y_true, y_pred, model_name, test_groups, cluster_labels)

    # ------------------------------------------------------------------
    # Reportes estructurados
    # ------------------------------------------------------------------
    def _build_report(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str,
                      test_groups: np.ndarray = None, cluster_labels: np.ndarray = None) -> dict:
        """Construye el reporte global / por área / por clúster e imprime resumen."""
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        report = {"model": model_name, "global": {}, "por_area": {}, "por_cluster": {}}

        report["global"] = self._compute_metrics(y_true_flat, y_pred_flat)
        self._print_metrics_dict(report["global"], f"{model_name} (Global)")

        if test_groups is not None:
            for area in ('Rural', 'Urbano'):
                mask = (test_groups == area)
                if np.any(mask):
                    report["por_area"][area] = self._compute_metrics(
                        y_true_flat[mask], y_pred_flat[mask])
                    self._print_metrics_dict(report["por_area"][area], f"{model_name} ({area})")

        if cluster_labels is not None:
            for cluster_id in sorted(np.unique(cluster_labels)):
                mask = (cluster_labels == cluster_id)
                if np.any(mask):
                    report["por_cluster"][str(int(cluster_id))] = self._compute_metrics(
                        y_true_flat[mask], y_pred_flat[mask])
            self._print_cluster_summary(report, model_name)

        return report

    def _print_cluster_summary(self, report: dict, model_name: str) -> None:
        """Imprime la tabla resumen por clúster (territorio funcional)."""
        print(f"\n--- Resumen por territorio funcional: {model_name} ---")
        print(f"{'Clúster':>8} | {'n':>8} | {'RMSE':>8} | {'MAE':>8} | {'R²':>7} | {'MBE':>8}")
        for cid, m in report["por_cluster"].items():
            print(f"{cid:>8} | {m['n']:>8} | {m['rmse']:>8.4f} | {m['mae']:>8.4f} "
                  f"| {m['r2']:>7.4f} | {m['mbe']:>+8.4f}")
        print("-" * 60)

    def evaluate_by_cluster(self, y_true: np.ndarray, y_pred: np.ndarray,
                            cluster_labels: np.ndarray, model_name: str = "Modelo") -> dict:
        """Evalúa el modelo por cada clúster LCA (territorio funcional).

        Args:
            y_true: Valores reales del target
            y_pred: Predicciones del modelo
            cluster_labels: Etiquetas de clúster LCA (0-6)
            model_name: Nombre del modelo para los prints

        Returns:
            Diccionario con métricas por clúster: {cluster_id: {metric: value}}
        """
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        results = {}
        unique_clusters = np.unique(cluster_labels)

        print(f"\n{'='*60}")
        print(f"  Evaluación por territorio funcional (LCA) — {model_name}")
        print(f"{'='*60}")

        for cluster_id in sorted(unique_clusters):
            mask = (cluster_labels == cluster_id)
            if np.any(mask):
                metrics = self._compute_metrics(y_true_flat[mask], y_pred_flat[mask])
                results[cluster_id] = metrics
                self._print_metrics_dict(metrics, f"{model_name} (Clúster {cluster_id})")

        return results

    def _print_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
        """Calcula e imprime métricas de regresión incluyendo sesgo medio.

        El Mean Bias Error (MBE) es la métrica clave de equidad: un valor
        negativo indica que el modelo subestima sistemáticamente a ese grupo;
        positivo indica sobreestimación. El RMSE por sí solo no detecta esto.
        """
        metrics = self._compute_metrics(y_true, y_pred)
        self._print_metrics_dict(metrics, model_name)

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calcula métricas de regresión y retorna como diccionario."""
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()

        return {
            "n": int(len(y_true_f)),
            "rmse": float(np.sqrt(mean_squared_error(y_true_f, y_pred_f))),
            "mae": float(mean_absolute_error(y_true_f, y_pred_f)),
            "r2": float(r2_score(y_true_f, y_pred_f)),
            "mbe": float(np.mean(y_pred_f - y_true_f)),
        }

    def _print_metrics_dict(self, metrics: dict, model_name: str):
        """Imprime métricas desde un diccionario."""
        mbe = metrics["mbe"]
        mbe_label = '(⚠ sobreestima)' if mbe > 1 else '(⚠ subestima)' if mbe < -1 else '(sin sesgo relevante)'

        print(f"\n--- Resultados en Test: {model_name} ---")
        print(f"n:    {metrics['n']}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE:  {metrics['mae']:.4f}")
        print(f"R²:   {metrics['r2']:.4f}")
        print(f"MBE:  {mbe:+.4f}  {mbe_label}")
        print("-" * 40)

    # ------------------------------------------------------------------
    # Persistencia de resultados
    # ------------------------------------------------------------------
    @staticmethod
    def save_report(report: dict, path: str) -> None:
        """Guarda un reporte de evaluación en JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Reporte de métricas guardado en {path}")

    @staticmethod
    def append_tabla4(report: dict, tag: str, path: str) -> None:
        """Añade las filas del reporte a un CSV tipo Tabla 4.

        Una fila por ámbito (Global, áreas y clústeres) con n / RMSE / MAE /
        R² / MBE: es el insumo directo de la Tabla 4 del artículo.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        file_exists = Path(path).exists()
        rows = [("Global", report["global"])]
        rows += [(f"Area {k}", v) for k, v in report.get("por_area", {}).items()]
        rows += [(f"Cluster {k}", v) for k, v in report.get("por_cluster", {}).items()]

        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["modelo", "tag", "ambito", "n", "rmse", "mae", "r2", "mbe"])
            for ambito, m in rows:
                writer.writerow([report["model"], tag, ambito, m["n"],
                                 f"{m['rmse']:.4f}", f"{m['mae']:.4f}",
                                 f"{m['r2']:.4f}", f"{m['mbe']:+.4f}"])
        print(f"Filas añadidas a {path}")
