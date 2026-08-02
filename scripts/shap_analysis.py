"""
Análisis SHAP segmentado por territorios funcionales (clústeres LCA).
Calcula SHAP values para los 14 modelos (7 XGBoost + 7 MLP) y genera
visualizaciones comparativas entre territorios.
"""
import argparse
import json
import shap
import torch
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import xgboost as xgb
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from saber_xai.config import config
from saber_xai.models.mlp_model import MLP
from saber_xai.models.xgb_model import XGBoostModel


def parse_args():
    parser = argparse.ArgumentParser(description="Análisis SHAP por territorios LCA")
    parser.add_argument("--models-dir", type=str, default="models",
                        help="Directorio con los modelos entrenados")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directorio con los datos por clúster")
    parser.add_argument("--output-dir", type=str, default="shap_results",
                        help="Directorio de salida para plots y resultados")
    parser.add_argument("--max-samples", type=int, default=10000,
                        help="Máximo de muestras para calcular SHAP")
    parser.add_argument("--clusters", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6],
                        help="IDs de clústeres a analizar")
    return parser.parse_args()


def load_cluster_data(data_dir: Path, cluster_id: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Carga datos de un clúster y retorna X, y, feature_names."""
    parquet_path = data_dir / f"clase_{cluster_id}.parquet"
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"No se encontró {parquet_path}")
    
    df = pl.read_parquet(parquet_path)
    
    # Separar features y target
    y = df.select("punt_global").to_numpy().flatten()
    
    # Eliminar columnas que no son features (si existen)
    cols_to_drop = ["punt_global"]
    for col in ["clase_lca", "prob_max_lca"]:
        if col in df.columns:
            cols_to_drop.append(col)
    
    X_df = df.drop(cols_to_drop)
    
    # One-hot encoding para categóricas
    cat_cols = [col for col in X_df.columns if X_df.schema[col] in (pl.String, pl.Categorical)]
    if cat_cols:
        X_df = X_df.to_dummies(cat_cols, drop_first=True)
    
    feature_names = X_df.columns
    X = X_df.to_numpy().astype(np.float32)
    
    return X, y, feature_names


def sample_data(X: np.ndarray, y: np.ndarray, max_samples: int, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Muestrea datos si exceden max_samples."""
    if len(X) <= max_samples:
        return X, y
    
    np.random.seed(random_state)
    indices = np.random.choice(len(X), max_samples, replace=False)
    return X[indices], y[indices]


def compute_shap_xgb(model_path: Path, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
    """Calcula SHAP values para modelo XGBoost."""
    print(f"  Cargando XGBoost desde {model_path}...")
    xgb_model = XGBoostModel()
    xgb_model.model = xgb.Booster()
    xgb_model.model.load_model(str(model_path))
    
    print(f"  Calculando SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(xgb_model.model)
    shap_values = explainer.shap_values(X)
    
    return shap_values


def compute_shap_mlp(model_path: Path, X: np.ndarray, input_dim: int, 
                     feature_names: List[str], max_background: int = 100) -> np.ndarray:
    """Calcula SHAP values para modelo MLP usando DeepExplainer."""
    print(f"  Cargando MLP desde {model_path}...")
    mlp = MLP(input_dim=input_dim)
    mlp.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    mlp.eval()
    
    # Background: subconjunto representativo
    background_size = min(max_background, len(X))
    background = torch.tensor(X[:background_size], dtype=torch.float32)
    
    print(f"  Calculando SHAP values (DeepExplainer)...")
    explainer = shap.DeepExplainer(mlp, background)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    shap_values = explainer.shap_values(X_tensor)
    
    # Squeeze para remover dimensión extra si existe
    if shap_values.ndim == 3:
        shap_values = shap_values.squeeze(-1)
    
    return shap_values


def plot_summary_comparison(all_shap_values: Dict[int, np.ndarray], 
                           all_X: Dict[int, np.ndarray],
                           feature_names: List[str],
                           output_dir: Path,
                           model_type: str):
    """Genera summary plots comparativos entre clústeres."""
    print("  Generando summary plots comparativos...")
    
    # Summary plot individual por clúster
    for cluster_id, shap_vals in all_shap_values.items():
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_vals, all_X[cluster_id], 
                         feature_names=feature_names, 
                         show=False,
                         plot_size=(12, 8))
        plt.title(f"SHAP Summary — Clúster {cluster_id} ({model_type})", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f"shap_summary_cluster_{cluster_id}_{model_type}.png", dpi=150)
        plt.close()
    
    # Bar plot comparativo: top-10 variables por clúster
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()
    
    for idx, cluster_id in enumerate(sorted(all_shap_values.keys())):
        if idx >= 7:
            break
        
        mean_abs_shap = np.abs(all_shap_values[cluster_id]).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-10:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        top_values = mean_abs_shap[top_indices]
        
        ax = axes[idx]
        ax.barh(range(len(top_features)), top_values, color=plt.cm.viridis(idx/7))
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features, fontsize=9)
        ax.set_xlabel("Mean |SHAP value|", fontsize=10)
        ax.set_title(f"Clúster {cluster_id}", fontsize=12, fontweight='bold')
        ax.invert_yaxis()
    
    # Ocultar subplot vacío si hay menos de 7 clústeres
    for idx in range(len(all_shap_values), 7):
        axes[idx].set_visible(False)
    
    plt.suptitle(f"Top-10 Variables por Territorio Funcional ({model_type})", 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / f"shap_comparison_top10_{model_type}.png", dpi=150)
    plt.close()


def export_importance_metrics(all_shap_values: Dict[int, np.ndarray],
                              feature_names: List[str],
                              output_dir: Path,
                              model_type: str):
    """Exporta métricas de importancia a CSV y JSON."""
    print("  Exportando métricas de importancia...")
    
    # Matriz: clústeres × features
    importance_matrix = {}
    for cluster_id, shap_vals in all_shap_values.items():
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        importance_matrix[cluster_id] = {
            feature_names[i]: float(mean_abs_shap[i]) 
            for i in range(len(feature_names))
        }
    
    # Exportar a JSON
    json_path = output_dir / f"shap_importance_{model_type}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(importance_matrix, f, indent=2, ensure_ascii=False)
    
    # Exportar a CSV (formato largo)
    csv_rows = []
    for cluster_id, features in importance_matrix.items():
        for feature, importance in features.items():
            csv_rows.append({
                "cluster_id": cluster_id,
                "feature": feature,
                "mean_abs_shap": importance
            })
    
    df = pl.DataFrame(csv_rows)
    csv_path = output_dir / f"shap_importance_{model_type}.csv"
    df.write_csv(csv_path)
    
    print(f"    → {json_path}")
    print(f"    → {csv_path}")
    
    return importance_matrix


def analyze_differential_importance(importance_xgb: Dict, importance_mlp: Dict,
                                    output_dir: Path):
    """Identifica variables con impacto diferencial entre territorios."""
    print("  Analizando importancia diferencial...")
    
    # Promediar XGBoost y MLP
    all_features = set()
    for cluster_data in importance_xgb.values():
        all_features.update(cluster_data.keys())
    
    differential_analysis = {}
    
    for feature in all_features:
        xgb_values = []
        mlp_values = []
        
        for cluster_id in sorted(importance_xgb.keys()):
            xgb_val = importance_xgb[cluster_id].get(feature, 0)
            mlp_val = importance_mlp[cluster_id].get(feature, 0)
            xgb_values.append(xgb_val)
            mlp_values.append(mlp_val)
        
        # Coeficiente de variación entre clústeres
        xgb_mean = np.mean(xgb_values)
        xgb_std = np.std(xgb_values)
        xgb_cv = xgb_std / xgb_mean if xgb_mean > 0 else 0
        
        differential_analysis[feature] = {
            "xgb_mean": float(xgb_mean),
            "xgb_std": float(xgb_std),
            "xgb_cv": float(xgb_cv),
            "xgb_by_cluster": xgb_values,
            "mlp_mean": float(np.mean(mlp_values)),
            "mlp_std": float(np.std(mlp_values)),
            "mlp_cv": float(np.std(mlp_values) / np.mean(mlp_values)) if np.mean(mlp_values) > 0 else 0,
            "mlp_by_cluster": mlp_values
        }
    
    # Exportar análisis
    json_path = output_dir / "differential_importance_analysis.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(differential_analysis, f, indent=2, ensure_ascii=False)
    
    # Top variables con mayor variación entre territorios
    top_differential = sorted(differential_analysis.items(), 
                             key=lambda x: x[1]["xgb_cv"], 
                             reverse=True)[:20]
    
    print("    Top-10 variables con mayor variación entre territorios (XGBoost):")
    for feature, metrics in top_differential[:10]:
        print(f"      {feature}: CV={metrics['xgb_cv']:.3f}")
    
    print(f"    → {json_path}")


def main():
    args = parse_args()
    
    models_dir = Path(args.models_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*70}")
    print("  ANÁLISIS SHAP POR TERRITORIOS FUNCIONALES (LCA)")
    print(f"{'='*70}")
    print(f"Modelos: {models_dir}")
    print(f"Datos: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Clústeres: {args.clusters}")
    print(f"Max samples: {args.max_samples:,}")
    
    # Cargar datos de todos los clústeres
    all_X = {}
    all_y = {}
    feature_names = None
    
    print(f"\n{'='*70}")
    print("  CARGANDO DATOS POR CLÚSTER")
    print(f"{'='*70}")
    
    for cluster_id in args.clusters:
        print(f"\nClúster {cluster_id}...")
        try:
            X, y, feature_names = load_cluster_data(data_dir, cluster_id)
            X_sampled, y_sampled = sample_data(X, y, args.max_samples)
            all_X[cluster_id] = X_sampled
            all_y[cluster_id] = y_sampled
            print(f"  ✓ {len(X_sampled):,} muestras (de {len(X):,} totales)")
            print(f"  ✓ {len(feature_names)} features")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    if not all_X:
        print("\n✗ No se cargaron datos. Abortando.")
        return
    
    # Análisis XGBoost
    print(f"\n{'='*70}")
    print("  ANÁLISIS SHAP — XGBOOST")
    print(f"{'='*70}")
    
    all_shap_xgb = {}
    for cluster_id in args.clusters:
        if cluster_id not in all_X:
            continue
        
        print(f"\nClúster {cluster_id}...")
        model_path = models_dir / f"xgb_cluster_{cluster_id}.json"
        
        if not model_path.exists():
            print(f"  ✗ Modelo no encontrado: {model_path}")
            continue
        
        try:
            shap_values = compute_shap_xgb(model_path, all_X[cluster_id], feature_names)
            all_shap_xgb[cluster_id] = shap_values
            print(f"  ✓ SHAP calculado: shape {shap_values.shape}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    if all_shap_xgb:
        print(f"\n{'='*70}")
        print("  VISUALIZACIONES XGBOOST")
        print(f"{'='*70}")
        
        plot_summary_comparison(all_shap_xgb, all_X, feature_names, output_dir, "XGBoost")
        importance_xgb = export_importance_metrics(all_shap_xgb, feature_names, output_dir, "XGBoost")
    else:
        importance_xgb = {}
        print("\n✗ No se calcularon SHAP values para XGBoost.")
    
    # Análisis MLP
    print(f"\n{'='*70}")
    print("  ANÁLISIS SHAP — MLP")
    print(f"{'='*70}")
    
    all_shap_mlp = {}
    input_dim = len(feature_names)
    
    for cluster_id in args.clusters:
        if cluster_id not in all_X:
            continue
        
        print(f"\nClúster {cluster_id}...")
        model_path = models_dir / f"mlp_cluster_{cluster_id}.pth"
        
        if not model_path.exists():
            print(f"  ✗ Modelo no encontrado: {model_path}")
            continue
        
        try:
            shap_values = compute_shap_mlp(model_path, all_X[cluster_id], input_dim, feature_names)
            all_shap_mlp[cluster_id] = shap_values
            print(f"  ✓ SHAP calculado: shape {shap_values.shape}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    if all_shap_mlp:
        print(f"\n{'='*70}")
        print("  VISUALIZACIONES MLP")
        print(f"{'='*70}")
        
        plot_summary_comparison(all_shap_mlp, all_X, feature_names, output_dir, "MLP")
        importance_mlp = export_importance_metrics(all_shap_mlp, feature_names, output_dir, "MLP")
    else:
        importance_mlp = {}
        print("\n✗ No se calcularon SHAP values para MLP.")
    
    # Análisis comparativo
    if importance_xgb and importance_mlp:
        print(f"\n{'='*70}")
        print("  ANÁLISIS DIFERENCIAL ENTRE TERRITORIOS")
        print(f"{'='*70}")
        
        analyze_differential_importance(importance_xgb, importance_mlp, output_dir)
    
    print(f"\n{'='*70}")
    print("  ANÁLISIS COMPLETADO")
    print(f"{'='*70}")
    print(f"Resultados guardados en: {output_dir}/")


if __name__ == "__main__":
    main()
