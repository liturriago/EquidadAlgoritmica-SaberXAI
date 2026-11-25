"""
Evaluación de calibración de los modelos por clúster LCA.

Responde dos preguntas para §5.2 del artículo:

1. **Mecanismo**: ¿el desplazamiento de la media de las predicciones (MBE) existe
   también en TRAIN y VAL, no solo en test? Si MBE_train ≈ MBE_val ≈ MBE_test,
   el sesgo es una propiedad de calibración de la función aprendida (no un
   artefacto del conjunto de evaluación ni de la partición).

2. **Remedio**: ¿la post-calibración por clúster (restar una constante estimada
   exclusivamente en VALIDACIÓN) elimina el sesgo en test? De ser así, el
   enfoque por clúster conserva su valor de diagnóstico territorial sin pagar
   el costo de equidad.

Uso:
  python scripts/evaluar_calibracion.py --config configs/lca_clusters.yaml \
      --models-dir models --model-type both --data-dir <dir parquet>

Requiere modelos entrenados (xgb_cluster_{id}.json + sidecar .meta.json,
mlp_cluster_{id}.pth) — p. ej. la salida del notebook 12 de Kaggle.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from saber_xai.config import config
from saber_xai.data.data_module import DataModule
from saber_xai.models.mlp_model import MLP


def parse_args():
    parser = argparse.ArgumentParser(description="Calibración de modelos por clúster")
    parser.add_argument("--config", type=str, default="configs/lca_clusters.yaml")
    parser.add_argument("--clusters", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6])
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--model-type", type=str, default="xgb",
                        choices=["xgb", "mlp", "both"])
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--allow-dummy", action="store_true")
    return parser.parse_args()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "n": int(len(y_true)),
        "mean_pred": float(y_pred.mean()),
        "mean_y": float(y_true.mean()),
        "mbe": float((y_pred - y_true).mean()),
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "r2": float(1.0 - ss_res / ss_tot),
    }


def _predict_xgb(model_path: Path, dmat) -> np.ndarray:
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    best_iter = None
    meta_path = Path(f"{model_path}.meta.json")
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            best_iter = json.load(f).get("best_iteration")
    if best_iter is not None:
        return booster.predict(dmat, iteration_range=(0, best_iter + 1))
    return booster.predict(dmat)


def _predict_mlp_splits(model_path: Path, dm: DataModule) -> dict:
    """Predice train/val/test con el MLP (mismo device que los DataLoaders)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlp = MLP(input_dim=dm.X_train.shape[1])
    mlp.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=True))
    mlp.to(device)
    mlp.eval()

    out = {}
    loaders = dm.get_dataloaders()
    for name, loader in zip(("train", "val", "test"), loaders):
        preds = []
        with torch.no_grad():
            for X_batch, _ in loader:
                preds.append(mlp(X_batch).cpu().numpy())
        out[name] = np.vstack(preds).flatten()
    return out


def evaluate_cluster(cluster_id: int, models_dir: Path, model_types: list) -> dict:
    print(f"\n{'=' * 60}\n  CALIBRACIÓN — CLÚSTER {cluster_id}\n{'=' * 60}")
    config.cluster_id = cluster_id
    dm = DataModule()
    dm.prepare_data()

    dtrain, dval, dtest = dm.get_dmatrices()
    y = {
        "train": dm.y_train.to_numpy().flatten(),
        "val": dm.y_val.to_numpy().flatten(),
        "test": dm.y_test.to_numpy().flatten(),
    }

    result = {"cluster_id": cluster_id, "models": {}}

    def _report(model_name: str, preds: dict) -> dict:
        rep = {split: _metrics(y[split], preds[split]) for split in ("train", "val", "test")}
        # Post-calibración: constante = MBE en VALIDACIÓN (sin mirar test)
        c = rep["val"]["mbe"]
        rep["post_calibracion"] = {
            "constante_val": float(c),
            "test": _metrics(y["test"], preds["test"] - c),
        }
        print(f"\n--- {model_name} (clúster {cluster_id}) ---")
        for split in ("train", "val", "test"):
            m = rep[split]
            print(f"{split:>5}: mean(pred) {m['mean_pred']:.2f} | mean(y) {m['mean_y']:.2f} "
                  f"| MBE {m['mbe']:+.2f} | RMSE {m['rmse']:.2f}")
        pc = rep["post_calibracion"]
        print(f"post-calibración (c={pc['constante_val']:+.2f} de VAL): "
              f"MBE_test {pc['test']['mbe']:+.2f} | RMSE {pc['test']['rmse']:.2f} "
              f"| R² {pc['test']['r2']:.4f}")
        return rep

    if "xgb" in model_types:
        xgb_path = models_dir / f"xgb_cluster_{cluster_id}.json"
        if xgb_path.exists():
            preds = {
                "train": _predict_xgb(xgb_path, dtrain),
                "val": _predict_xgb(xgb_path, dval),
                "test": _predict_xgb(xgb_path, dtest),
            }
            result["models"]["xgb"] = _report("XGBoost", preds)
        else:
            print(f"  (sin modelo XGBoost: {xgb_path})")

    if "mlp" in model_types:
        mlp_path = models_dir / f"mlp_cluster_{cluster_id}.pth"
        if mlp_path.exists():
            preds = _predict_mlp_splits(mlp_path, dm)
            result["models"]["mlp"] = _report("MLP", preds)
        else:
            print(f"  (sin modelo MLP: {mlp_path})")

    return result


def main():
    args = parse_args()
    config.load_from_yaml(args.config)
    if args.data_dir:
        config.data_dir = args.data_dir
        print(f"data_dir sobrescrito por CLI: {config.data_dir}")
    if args.allow_dummy:
        config.allow_dummy_data = True

    model_types = ["xgb", "mlp"] if args.model_type == "both" else [args.model_type]
    models_dir = Path(args.models_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for cluster_id in args.clusters:
        results.append(evaluate_cluster(cluster_id, models_dir, model_types))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = results_dir / "calibracion_mbe.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nCalibración guardada en {out_path}")

    print(f"\n{'=' * 78}\n  RESUMEN — MBE en train / val / test y tras post-calibración\n{'=' * 78}")
    print(f"{'Clúster':>7} {'modelo':>6} | {'MBE train':>9} {'MBE val':>8} {'MBE test':>8} "
          f"{'| MBE test corr.':>15} {'RMSE corr.':>10}")
    for r in results:
        for mt, rep in r["models"].items():
            pc = rep["post_calibracion"]["test"]
            print(f"{r['cluster_id']:>7} {mt:>6} | {rep['train']['mbe']:>+9.2f} "
                  f"{rep['val']['mbe']:>+8.2f} {rep['test']['mbe']:>+8.2f} "
                  f"{pc['mbe']:>+15.2f} {pc['rmse']:>10.2f}")


if __name__ == "__main__":
    main()
