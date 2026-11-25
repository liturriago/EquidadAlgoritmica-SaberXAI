import polars as pl
import polars.selectors as cs
import numpy as np
import gc
import json
import xgboost as xgb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import TargetEncoder, StandardScaler
from pathlib import Path

from saber_xai.config import config

class ICFESDataset(Dataset):
    """Dataset de PyTorch para los datos del ICFES.

    Soporta dos modos de operación:
    - ``device='cpu'`` (default): tensores en RAM, transferencia por batch.
    - ``device='cuda'``: tensores pre-cargados en VRAM una sola vez. Elimina
      todas las transferencias CPU→GPU durante el entrenamiento. Requiere que
      el dataset completo quepa en la VRAM disponible.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, device: str = "cpu"):
        # Polars devuelve arrays read-only; .copy() garantiza un buffer escribible
        # antes de pasarlo a from_numpy (evita el UserWarning de PyTorch).
        self.X = torch.from_numpy(np.ascontiguousarray(X).copy()).to(device)
        self.y = torch.from_numpy(np.ascontiguousarray(y.reshape(-1, 1)).copy()).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DataModule:
    """
    Fábrica y administrador de datos para el proyecto SaberXAI.
    Centraliza la carga, división, imputación, codificación y escalado usando Polars.

    Modos de operación (resueltos desde la configuración):

    - ``cluster`` — ``config.cluster_id`` + ``config.data_dir``: carga
      ``clase_{cluster_id}.parquet`` (un territorio funcional LCA).
    - ``global``  — solo ``config.data_dir``: carga la unión de los
      ``clase_*.parquet`` replicando el split por clúster (misma semilla y
      proporciones 70/15/15), de modo que el modelo global se entrena y evalúa
      sobre **exactamente las mismas filas de test** que los modelos por
      clúster (comparación válida de MBE/R²/RMSE).
    - ``file``    — ``config.data_path``: un único parquet (modo legacy).

    Comportamientos adicionales:
    - ``config.territory_indicators=True`` añade indicadoras one-hot del clúster
      LCA a las features (solo modo ``global``; drop_first=True).
    - ``self.test_cluster_labels`` alinea cada fila de test con su clúster LCA
      para la evaluación por territorio (None en modo ``file`` sin columna).
    - ``self.split_meta`` documenta tamaños de split, medias del target y
      lista de features (reproducibilidad y diagnóstico del MBE).
    - La carga es estricta por defecto: si falta el parquet se lanza
      ``FileNotFoundError``. Los datos dummy solo se generan si
      ``config.allow_dummy_data=True`` (para tests de integración).
    """

    TERRITORY_COL = "clase_lca"

    def __init__(self):
        self.config = config
        self.target_encoder = TargetEncoder(target_type="continuous", random_state=config.random_state)
        self.scaler = StandardScaler()

        # Contenedores para los sets procesados
        self.X_train, self.X_val, self.X_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None
        self.test_area = None

        # Metadatos del split (modo, etiquetas de territorio, trazabilidad)
        self.mode = None
        self.cluster_id = None
        self.test_cluster_labels = None
        self.split_meta = {}

    # ------------------------------------------------------------------
    # Carga de datos
    # ------------------------------------------------------------------
    def _resolve_mode(self) -> str:
        """Determina el modo de carga a partir de la configuración."""
        if self.config.cluster_id is not None and self.config.data_dir is not None:
            return "cluster"
        if self.config.data_dir is not None:
            return "global"
        return "file"

    def _load_frames(self) -> list[tuple]:
        """Carga los parquet crudos como lista de pares (cluster_id | None, df).

        En modo ``global`` cada frame conserva su ``cluster_id`` para poder
        replicar el split por clúster antes de unir los conjuntos.
        """
        self.mode = self._resolve_mode()
        self.cluster_id = self.config.cluster_id
        try:
            if self.mode == "cluster":
                data_path = Path(self.config.data_dir) / f"clase_{self.config.cluster_id}.parquet"
                print(f"Cargando datos del clúster LCA {self.config.cluster_id}: {data_path}")
                return [(self.config.cluster_id, pl.read_parquet(data_path))]

            if self.mode == "global":
                data_dir = Path(self.config.data_dir)
                paths = sorted(data_dir.glob("clase_*.parquet"))
                if not paths:
                    raise FileNotFoundError(f"No se encontraron clase_*.parquet en {data_dir}")
                print(f"Cargando {len(paths)} clústeres LCA desde {data_dir} (modo global)")
                frames = []
                for p in paths:
                    cid = int(p.stem.split("_")[1])
                    frames.append((cid, pl.read_parquet(p)))
                return frames

            # Modo legacy: un único archivo
            print(f"Cargando dataset único: {self.config.data_path}")
            return [(None, pl.read_parquet(Path(self.config.data_path)))]

        except Exception as e:
            if not self.config.allow_dummy_data:
                raise
            print(f"Advertencia: Error al cargar Parquet: {e}. Generando datos dummy...")
            return [(self.config.cluster_id, self._generate_dummy_data())]

    def _split_frame(self, df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Divide un frame 70/15/15 tras barajar con la semilla del proyecto.

        El orden de las operaciones replica exactamente el pipeline histórico
        (barajado con ``seed=random_state``; test = primer 15 %, val = siguiente
        15 %, train = resto), para mantener la comparabilidad con los modelos
        ya entrenados y el diagnóstico del MBE.
        """
        df = df.sample(fraction=1.0, shuffle=True, seed=self.config.random_state)

        total_rows = len(df)
        test_size = int(total_rows * 0.15)
        val_size = int(total_rows * 0.15)
        train_size = total_rows - test_size - val_size

        df_test = df.head(test_size)
        df_val = df.slice(test_size, val_size)
        df_train = df.tail(train_size)
        return df_train, df_val, df_test

    # ------------------------------------------------------------------
    # Pipeline completo
    # ------------------------------------------------------------------
    def prepare_data(self) -> None:
        """
        Flujo completo de ingeniería de datos protegiendo la integridad científica.
        """
        frames = self._load_frames()

        # 1. Split por frame (por clúster en modo global/cluster) y unión.
        #    Así el modo global evalúa sobre las MISMAS filas de test que los
        #    modelos por clúster.
        trains, vals, tests = [], [], []
        cluster_rows = {}
        for cid, df in frames:
            df_train, df_val, df_test = self._split_frame(df)
            if cid is not None:
                df_train = df_train.with_columns(pl.lit(cid, dtype=pl.Int32).alias(self.TERRITORY_COL))
                df_val = df_val.with_columns(pl.lit(cid, dtype=pl.Int32).alias(self.TERRITORY_COL))
                df_test = df_test.with_columns(pl.lit(cid, dtype=pl.Int32).alias(self.TERRITORY_COL))
                cluster_rows[cid] = {
                    "train": len(df_train), "val": len(df_val), "test": len(df_test)
                }
            trains.append(df_train)
            vals.append(df_val)
            tests.append(df_test)
            del df

        self.X_train = pl.concat(trains)
        self.X_val = pl.concat(vals)
        self.X_test = pl.concat(tests)
        del trains, vals, tests
        gc.collect()

        self.y_test = self.X_test.select(pl.col(self.config.target_col))
        self.X_test = self.X_test.drop(self.config.target_col)

        self.y_val = self.X_val.select(pl.col(self.config.target_col))
        self.X_val = self.X_val.drop(self.config.target_col)

        self.y_train = self.X_train.select(pl.col(self.config.target_col))
        self.X_train = self.X_train.drop(self.config.target_col)

        # Guardar la columna de área para la evaluación de equidad en el Test set
        if 'cole_area_ubicacion' in self.X_test.columns:
            self.test_area = self.X_test.get_column('cole_area_ubicacion').to_numpy()

        # 2. Etiquetas de territorio del test (antes de transformar features)
        if self.TERRITORY_COL in self.X_test.columns:
            self.test_cluster_labels = self.X_test.get_column(self.TERRITORY_COL).to_numpy()

        # 3. Indicadoras de territorio o eliminación de la columna cruda
        if self.TERRITORY_COL in self.X_train.columns:
            if self.config.territory_indicators and self.mode == "global":
                self._add_territory_indicators()
            else:
                self.X_train = self.X_train.drop(self.TERRITORY_COL)
                self.X_val = self.X_val.drop(self.TERRITORY_COL)
                self.X_test = self.X_test.drop(self.TERRITORY_COL)

        # 4. Imputación de valores nulos (Mediana)
        # Se calcula en train y se aplica en todos para evitar sesgos
        num_cols = self.X_train.select(cs.numeric()).columns
        for col in num_cols:
            median_val = self.X_train.get_column(col).median()
            if median_val is not None:
                self.X_train = self.X_train.with_columns(pl.col(col).fill_null(median_val))
                self.X_val = self.X_val.with_columns(pl.col(col).fill_null(median_val))
                self.X_test = self.X_test.with_columns(pl.col(col).fill_null(median_val))

        # 5. Target Encoding (Específico para Municipio) — Solo si está configurado
        cat_to_encode = self.config.cat_col_to_encode
        if cat_to_encode is not None and cat_to_encode in self.X_train.columns:
            # Fit solo en Train para prevenir data leakage
            X_train_cat = self.X_train.select(cat_to_encode).to_numpy()
            y_train_np = self.y_train.to_numpy().flatten()
            encoded_train = self.target_encoder.fit_transform(X_train_cat, y_train_np)
            self.X_train = self.X_train.with_columns(pl.Series(cat_to_encode, encoded_train.flatten()))

            X_val_cat = self.X_val.select(cat_to_encode).to_numpy()
            encoded_val = self.target_encoder.transform(X_val_cat)
            self.X_val = self.X_val.with_columns(pl.Series(cat_to_encode, encoded_val.flatten()))

            X_test_cat = self.X_test.select(cat_to_encode).to_numpy()
            encoded_test = self.target_encoder.transform(X_test_cat)
            self.X_test = self.X_test.with_columns(pl.Series(cat_to_encode, encoded_test.flatten()))
        elif cat_to_encode is not None:
            print(f"Advertencia: cat_col_to_encode='{cat_to_encode}' no encontrada en los datos. Saltando Target Encoding.")

        # 6. One-Hot Encoding para categorías restantes
        remaining_cats = [col for col in self.X_train.columns if self.X_train.schema[col] in (pl.String, pl.Categorical)]
        if len(remaining_cats) > 0:
            self.X_train = self.X_train.to_dummies(remaining_cats, drop_first=True)
            self.X_val = self.X_val.to_dummies(remaining_cats, drop_first=True)
            self.X_test = self.X_test.to_dummies(remaining_cats, drop_first=True)

            # Alinear columnas entre Train, Val y Test
            train_cols = self.X_train.columns

            def align_cols(df_target: pl.DataFrame, base_cols: list) -> pl.DataFrame:
                for col in base_cols:
                    if col not in df_target.columns:
                        df_target = df_target.with_columns(pl.lit(0).alias(col))
                return df_target.select(base_cols)

            self.X_val = align_cols(self.X_val, train_cols)
            self.X_test = align_cols(self.X_test, train_cols)

        # 7. Escalado Estándar (Z-Score)
        cols = self.X_train.columns
        X_train_np = self.X_train.to_numpy()
        self.X_train = None  # Liberar Polars antes de crear la copia escalada
        self.X_train = pl.DataFrame(self.scaler.fit_transform(X_train_np), schema=cols, orient="row")
        del X_train_np

        X_val_np = self.X_val.to_numpy()
        self.X_val = None
        self.X_val = pl.DataFrame(self.scaler.transform(X_val_np), schema=cols, orient="row")
        del X_val_np

        X_test_np = self.X_test.to_numpy()
        self.X_test = None
        self.X_test = pl.DataFrame(self.scaler.transform(X_test_np), schema=cols, orient="row")
        del X_test_np
        gc.collect()

        # 8. Asegurar tipo Float32 para PyTorch
        self.X_train = self.X_train.cast(pl.Float32)
        self.X_val = self.X_val.cast(pl.Float32)
        self.X_test = self.X_test.cast(pl.Float32)

        self.y_train = self.y_train.cast(pl.Float32)
        self.y_val = self.y_val.cast(pl.Float32)
        self.y_test = self.y_test.cast(pl.Float32)

        # 9. Metadatos de trazabilidad del split
        self.split_meta = {
            "mode": self.mode,
            "cluster_id": self.cluster_id,
            "seed": self.config.random_state,
            "territory_indicators": bool(
                self.config.territory_indicators and self.mode == "global"
            ),
            "n_train": len(self.X_train),
            "n_val": len(self.X_val),
            "n_test": len(self.X_test),
            "y_mean_train": float(self.y_train.to_numpy().mean()),
            "y_mean_val": float(self.y_val.to_numpy().mean()),
            "y_mean_test": float(self.y_test.to_numpy().mean()),
            "n_features": len(self.X_train.columns),
            "feature_names": list(self.X_train.columns),
            "cluster_rows": cluster_rows or None,
        }

    def _add_territory_indicators(self) -> None:
        """Añade indicadoras one-hot del clúster LCA a las features (drop_first).

        Solo tiene sentido en modo ``global``: identifica el territorio
        funcional de cada estudiante para el modelo "global + indicadoras".
        La columna cruda ``clase_lca`` se elimina de las features.
        """
        levels = sorted(self.X_train.get_column(self.TERRITORY_COL).unique().to_list())
        keep = levels[1:] if len(levels) > 1 else []
        for lvl in keep:
            alias = f"{self.TERRITORY_COL}_{lvl}"
            self.X_train = self.X_train.with_columns(
                (pl.col(self.TERRITORY_COL) == lvl).cast(pl.Float32).alias(alias)
            )
            self.X_val = self.X_val.with_columns(
                (pl.col(self.TERRITORY_COL) == lvl).cast(pl.Float32).alias(alias)
            )
            self.X_test = self.X_test.with_columns(
                (pl.col(self.TERRITORY_COL) == lvl).cast(pl.Float32).alias(alias)
            )
        self.X_train = self.X_train.drop(self.TERRITORY_COL)
        self.X_val = self.X_val.drop(self.TERRITORY_COL)
        self.X_test = self.X_test.drop(self.TERRITORY_COL)
        print(f"Indicadoras de territorio añadidas: {len(keep)} columnas ({self.TERRITORY_COL}_*)")

    # ------------------------------------------------------------------
    # Salidas para los modelos
    # ------------------------------------------------------------------
    def get_dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Retorna objetos DataLoader listos para el entrenamiento de la MLP.

        Si CUDA está disponible, los tres datasets se pre-cargan completos en
        VRAM (GPU-resident tensors). Esto elimina todas las transferencias
        CPU→GPU durante el entrenamiento a cambio de consumo fijo de VRAM.

        Restricciones del modo GPU-resident:
        - ``num_workers`` debe ser 0: los tensores CUDA no pueden ser
          compartidos entre procesos worker (spawn/fork no lo soportan).
        - ``pin_memory`` debe ser False: los tensores ya están en VRAM,
          no en memoria paginable de CPU.
        """
        cuda_available = torch.cuda.is_available()
        dataset_device = "cuda" if cuda_available else "cpu"

        # num_workers > 0 es incompatible con tensores CUDA en el dataset;
        # en CPU se puede usar workers pero para datos tabulares en RAM
        # el overhead de spawn en Windows/Kaggle suele no compensar.
        num_workers = 0
        pin_memory = False  # False si los tensores ya están en GPU

        train_ds = ICFESDataset(self.X_train.to_numpy(), self.y_train.to_numpy(), device=dataset_device)
        val_ds = ICFESDataset(self.X_val.to_numpy(), self.y_val.to_numpy(), device=dataset_device)
        test_ds = ICFESDataset(self.X_test.to_numpy(), self.y_test.to_numpy(), device=dataset_device)

        if cuda_available:
            print(f"[DataLoader] Datos pre-cargados en GPU ({dataset_device}). "
                  "Transferencias CPU→GPU eliminadas.")

        return (
            DataLoader(train_ds, batch_size=self.config.mlp_batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=pin_memory),
            DataLoader(val_ds, batch_size=self.config.mlp_batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=pin_memory),
            DataLoader(test_ds, batch_size=self.config.mlp_batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=pin_memory),
        )

    def get_dmatrices(self) -> tuple[xgb.DMatrix, xgb.DMatrix, xgb.DMatrix]:
        """Retorna objetos DMatrix optimizados para XGBoost."""
        return (
            xgb.DMatrix(self.X_train.to_numpy(), label=self.y_train.to_numpy(), feature_names=self.X_train.columns),
            xgb.DMatrix(self.X_val.to_numpy(), label=self.y_val.to_numpy(), feature_names=self.X_val.columns),
            xgb.DMatrix(self.X_test.to_numpy(), label=self.y_test.to_numpy(), feature_names=self.X_test.columns)
        )

    def save_split_meta(self, path: str) -> None:
        """Persiste los metadatos del split en JSON (trazabilidad del experimento)."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.split_meta, f, indent=2, ensure_ascii=False)
        print(f"Metadatos del split guardados en {path}")

    def _generate_dummy_data(self) -> pl.DataFrame:
        """Generador de datos sintéticos para pruebas de integración."""
        np.random.seed(self.config.random_state)
        n = 1000

        cat_col_name = self.config.cat_col_to_encode if self.config.cat_col_to_encode else "dummy_category"

        data = {
            cat_col_name: np.random.choice(['MUN_A', 'MUN_B', 'MUN_C'], n),
            'cole_area_ubicacion': np.random.choice(['Rural', 'Urbano'], n),
            'cole_naturaleza': np.random.choice(['OFICIAL', 'NO OFICIAL'], n),
            'fami_estrato': np.random.randint(1, 7, n),
            'estu_edad': np.random.uniform(15, 60, n),
            self.config.target_col: np.random.uniform(0, 500, n)
        }
        return pl.DataFrame(data)
