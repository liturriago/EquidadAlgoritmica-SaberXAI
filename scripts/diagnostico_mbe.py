"""
Diagnóstico del MBE (Mean Bias Error) por clúster LCA — prueba decisiva H1 vs H5.

Resuelve la contradicción abierta del entregable 3: si la partición fuera
intercambiable el MBE debería ser ≈ 0, pero se observó −10,41 en el clúster 6.

Hipótesis:
  H1 — el shuffle no fue efectivo en la ejecución original (el test no es
       intercambiable con train): las medias de y difieren entre splits
       aproximadamente en la magnitud del MBE.
  H5 — el sesgo vive en la etapa de predicción/evaluación (alineación X/y):
       las medias de y son estables entre splits, pero mean(pred) queda
       desplazada respecto de mean(y_test).

Uso:
  python scripts/diagnostico_mbe.py --config configs/lca_clusters.yaml \
      --models-dir models --model-type xgb

Sin modelos guardados hace la mitad "H1" (medias por split); con modelos
(xgb_cluster_{id}.json / mlp_cluster_{id}.pth) completa el veredicto.
Los modelos XGBoost leen la mejor ronda desde el sidecar
``{modelo}.meta.json`` si existe (ver train.py).
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
    parser = argparse.ArgumentParser(description="Diagnóstico del MBE por clúster LCA")
    parser.add_argument("--config", type=str, default="configs/lca_clusters.yaml")
    parser.add_argument("--clusters", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6])
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--model-type", type=str, default="xgb",
                        choices=["xgb", "mlp", "both"])
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Sobreescribe config.data_dir (carpeta con clase_*.parquet)")
    parser.add_argument("--allow-dummy", action="store_true")
    return parser.parse_args()


def _veredicto(delta_split: float, mbe: float) -> str:
    """Lectura del resultado según las hipótesis H1/H5 del handoff."""
    if abs(mbe) <= 1.0:
        return "SIN SESGO RELEVANTE (|MBE| ≤ 1): el MBE histórico no se reproduce"
    if abs(delta_split) >= 0.5 * abs(mbe) and np.sign(delta_split) == np.sign(mbe):
        return (f"H1 FAVORECIDA: mean(y_test) difiere de mean(y_train) en "
                f"{delta_split:+.2f} ≈ magnitud del MBE ({mbe:+.2f})")
    if abs(delta_split) < 0.5 * abs(mbe):
        return (f"H5 FAVORECIDA: splits intercambiables (Δmean(y) = {delta_split:+.2f}) "
                f"pero el MBE es {mbe:+.2f} → sesgo en predicción/evaluación")
    return "INCONCLUSO: revisar Δmean(y) y MBE manualmente"


def _predict_xgb(model_path: Path, dtest) -> np.ndarray:
    """Predice con el mejor iteration_range disponible (sidecar .meta.json)."""
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(str(model_path))

    best_iter = None
    meta_path = Path(f"{model_path}.meta.json")
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            best_iter = json.load(f).get("best_iteration")

    if best_iter is not None:
        print(f"  XGBoost: predicción con best_iteration={best_iter}")
        return booster.predict(dtest, iteration_range=(0, best_iter + 1))
    print("  XGBoost: sin sidecar .meta.json → predicción con todas las rondas")
    return booster.predict(dtest)


def _predict_mlp(model_path: Path, data_module: DataModule) -> np.ndarray:
    """Predice con los pesos del MLP sobre el DataLoader de test.

    El modelo se carga en el mismo device que los DataLoaders (que pre-cargan
    los tensores en GPU cuando CUDA está disponible).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = data_module.X_train.shape[1]
    mlp = MLP(input_dim=input_dim)
    mlp.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=True))
    mlp.to(device)
    mlp.eval()

    _, _, test_loader = data_module.get_dataloaders()
    preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            preds.append(mlp(X_batch).cpu().numpy())
    return np.vstack(preds).flatten()


def diagnose_cluster(cluster_id: int, models_dir: Path, model_types: list,
                     results_dir: Path) -> dict:
    print(f"\n{'=' * 60}\n  DIAGNÓSTICO MBE — CLÚSTER {cluster_id}\n{'=' * 60}")
    config.cluster_id = cluster_id
    dm = DataModule()
    dm.prepare_data()

    y_train = dm.y_train.to_numpy().flatten()
    y_val = dm.y_val.to_numpy().flatten()
    y_test = dm.y_test.to_numpy().flatten()
    dtest = dm.get_dmatrices()[2]

    result = {
        "cluster_id": cluster_id,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "y_mean_train": float(y_train.mean()),
        "y_mean_val": float(y_val.mean()),
        "y_mean_test": float(y_test.mean()),
        "y_std_test": float(y_test.std()),
        "models": {},
    }
    result["delta_split_test_minus_train"] = result["y_mean_test"] - result["y_mean_train"]

    print(f"n: train {result['n_train']:,} | val {result['n_val']:,} | test {result['n_test']:,}")
    print(f"mean(y): train {result['y_mean_train']:.2f} | val {result['y_mean_val']:.2f} "
          f"| test {result['y_mean_test']:.2f} | Δ(test−train) {result['delta_split_test_minus_train']:+.2f}")

    # Chequeo de alineación X/y del DMatrix de test (base de la hipótesis H5)
    labels_match = bool(np.allclose(dtest.get_label(), y_test, equal_nan=True))
    result["dtest_labels_match_y_test"] = labels_match
    print(f"Alineación dtest.label == y_test: {'OK' if labels_match else '¡DESCUADRE!'}")

    if "xgb" in model_types:
        xgb_path = models_dir / f"xgb_cluster_{cluster_id}.json"
        if xgb_path.exists():
            pred = _predict_xgb(xgb_path, dtest)
            mbe = float(pred.mean() - y_test.mean())
            from sklearn.metrics import r2_score, mean_squared_error
            result["models"]["xgb"] = {
                "mean_pred": float(pred.mean()),
                "mbe": mbe,
                "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                "r2": float(r2_score(y_test, pred)),
                "veredicto": _veredicto(result["delta_split_test_minus_train"], mbe),
            }
            m = result["models"]["xgb"]
            print(f"XGBoost: mean(pred) {m['mean_pred']:.2f} | MBE {m['mbe']:+.2f} "
                  f"| RMSE {m['rmse']:.2f} | R² {m['r2']:.4f}")
            print(f"Veredicto: {m['veredicto']}")
        else:
            print(f"  (sin modelo XGBoost: {xgb_path})")

    if "mlp" in model_types:
        mlp_path = models_dir / f"mlp_cluster_{cluster_id}.pth"
        if mlp_path.exists():
            pred = _predict_mlp(mlp_path, dm)
            mbe = float(pred.mean() - y_test.mean())
            from sklearn.metrics import r2_score, mean_squared_error
            result["models"]["mlp"] = {
                "mean_pred": float(pred.mean()),
                "mbe": mbe,
                "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                "r2": float(r2_score(y_test, pred)),
                "veredicto": _veredicto(result["delta_split_test_minus_train"], mbe),
            }
            m = result["models"]["mlp"]
            print(f"MLP: mean(pred) {m['mean_pred']:.2f} | MBE {m['mbe']:+.2f} "
                  f"| RMSE {m['rmse']:.2f} | R² {m['r2']:.4f}")
            print(f"Veredicto: {m['veredicto']}")
        else:
            print(f"  (sin modelo MLP: {mlp_path})")

    return result


def main():
    args = parse_args()
    config.load_from_yaml(args.config)
    if args.allow_dummy:
        config.allow_dummy_data = True

    model_types = ["xgb", "mlp"] if args.model_type == "both" else [args.model_type]
    models_dir = Path(args.models_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for cluster_id in args.clusters:
        results.append(diagnose_cluster(cluster_id, models_dir, model_types, results_dir))
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    out_path = results_dir / "diagnostico_mbe.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDiagnóstico guardado en {out_path}")

    print(f"\n{'=' * 60}\n  RESUMEN\n{'=' * 60}")
    print(f"{'Clúster':>8} | {'Δmean(y)':>9} | {'MBE xgb':>9} | {'MBE mlp':>9}")
    for r in results:
        mx = r["models"].get("xgb", {}).get("mbe")
        mm = r["models"].get("mlp", {}).get("mbe")
        print(f"{r['cluster_id']:>8} | {r['delta_split_test_minus_train']:>+9.2f} "
              f"| {('%+.2f' % mx) if mx is not None else '—':>9} "
              f"| {('%+.2f' % mm) if mm is not None else '—':>9}")


if __name__ == "__main__":
    main()
