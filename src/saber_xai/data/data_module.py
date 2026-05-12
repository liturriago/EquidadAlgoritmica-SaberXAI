import polars as pl
import polars.selectors as cs
import numpy as np
import gc
import xgboost as xgb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import TargetEncoder, StandardScaler

from saber_xai.config import config

class ICFESDataset(Dataset):
    """Dataset de PyTorch para los datos del ICFES.
    
    Los arrays numpy se guardan en disco y se convierten a tensor de forma
    lazy en __getitem__ para evitar duplicar toda la RAM con tensores.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X  # numpy float32, NO se convierte a tensor aquí
        self.y = y.reshape(-1)  # 1D para indexar por ítem

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # La conversión a tensor ocurre por ítem, no sobre todo el dataset
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(0)
        )


class DataModule:
    """
    Fábrica y administrador de datos para el proyecto SaberXAI.
    Centraliza la carga, división, imputación, codificación y escalado usando Polars.
    """
    def __init__(self):
        self.config = config
        self.target_encoder = TargetEncoder(target_type="continuous", random_state=config.random_state)
        self.scaler = StandardScaler()
        
        # Contenedores para los sets procesados
        self.X_train, self.X_val, self.X_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None
        self.test_area = None

    def prepare_data(self) -> None:
        """
        Flujo completo de ingeniería de datos protegiendo la integridad científica.
        """
        try:
            # Carga desde Parquet serializado
            df = pl.read_parquet(self.config.data_path)
        except Exception as e:
            print(f"Advertencia: Error al cargar Parquet: {e}. Generando datos dummy...")
            df = self._generate_dummy_data()

        # 1. Separación de Target y Features y Shuffle
        df = df.sample(fraction=1.0, shuffle=True, seed=self.config.random_state)
        
        # 2. División de conjuntos (70% Train, 15% Val, 15% Test)
        total_rows = len(df)
        test_size = int(total_rows * 0.15)
        val_size = int(total_rows * 0.15)
        train_size = total_rows - test_size - val_size
        
        df_test = df.head(test_size)
        df_val = df.slice(test_size, val_size)
        df_train = df.tail(train_size)
        
        self.y_test = df_test.select(pl.col(self.config.target_col))
        self.X_test = df_test.drop(self.config.target_col)
        
        self.y_val = df_val.select(pl.col(self.config.target_col))
        self.X_val = df_val.drop(self.config.target_col)
        
        self.y_train = df_train.select(pl.col(self.config.target_col))
        self.X_train = df_train.drop(self.config.target_col)

        # Guardar la columna de área para la evaluación de equidad en el Test set
        if 'cole_area_ubicacion' in self.X_test.columns:
            self.test_area = self.X_test.get_column('cole_area_ubicacion').to_numpy()

        # Liberar el DataFrame original y los splits intermedios de memoria
        del df, df_train, df_val, df_test
        gc.collect()

        # 3. Imputación de valores nulos (Mediana)
        # Se calcula en train y se aplica en todos para evitar sesgos
        num_cols = self.X_train.select(cs.numeric()).columns
        for col in num_cols:
            median_val = self.X_train.get_column(col).median()
            if median_val is not None:
                self.X_train = self.X_train.with_columns(pl.col(col).fill_null(median_val))
                self.X_val = self.X_val.with_columns(pl.col(col).fill_null(median_val))
                self.X_test = self.X_test.with_columns(pl.col(col).fill_null(median_val))

        # 4. Target Encoding (Específico para Municipio)
        cat_to_encode = self.config.cat_col_to_encode # ej. 'cole_mcpio_ubicacion'
        if cat_to_encode in self.X_train.columns:
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

        # 5. One-Hot Encoding para categorías restantes
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

        # 6. Escalado Estándar (Z-Score)
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

        # 7. Asegurar tipo Float32 para PyTorch
        self.X_train = self.X_train.cast(pl.Float32)
        self.X_val = self.X_val.cast(pl.Float32)
        self.X_test = self.X_test.cast(pl.Float32)
        
        self.y_train = self.y_train.cast(pl.Float32)
        self.y_val = self.y_val.cast(pl.Float32)
        self.y_test = self.y_test.cast(pl.Float32)

    def get_dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Retorna objetos DataLoader listos para el entrenamiento de la MLP."""
        train_ds = ICFESDataset(self.X_train.to_numpy(), self.y_train.to_numpy())
        val_ds = ICFESDataset(self.X_val.to_numpy(), self.y_val.to_numpy())
        test_ds = ICFESDataset(self.X_test.to_numpy(), self.y_test.to_numpy())

        return (
            DataLoader(train_ds, batch_size=self.config.mlp_batch_size, shuffle=True, num_workers=4, pin_memory=True),
            DataLoader(val_ds, batch_size=self.config.mlp_batch_size, shuffle=False, num_workers=4, pin_memory=True),
            DataLoader(test_ds, batch_size=self.config.mlp_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        )

    def get_dmatrices(self) -> tuple[xgb.DMatrix, xgb.DMatrix, xgb.DMatrix]:
        """Retorna objetos DMatrix optimizados para XGBoost."""
        return (
            xgb.DMatrix(self.X_train.to_numpy(), label=self.y_train.to_numpy(), feature_names=self.X_train.columns),
            xgb.DMatrix(self.X_val.to_numpy(), label=self.y_val.to_numpy(), feature_names=self.X_val.columns),
            xgb.DMatrix(self.X_test.to_numpy(), label=self.y_test.to_numpy(), feature_names=self.X_test.columns)
        )

    def _generate_dummy_data(self) -> pl.DataFrame:
        """Generador de datos sintéticos para pruebas de integración."""
        np.random.seed(self.config.random_state)
        n = 1000
        data = {
            self.config.cat_col_to_encode: np.random.choice(['MUN_A', 'MUN_B', 'MUN_C'], n),
            'cole_area_ubicacion': np.random.choice(['Rural', 'Urbano'], n),
            'cole_naturaleza': np.random.choice(['OFICIAL', 'NO OFICIAL'], n),
            'fami_estrato': np.random.randint(1, 7, n),
            'estu_edad': np.random.uniform(15, 60, n),
            self.config.target_col: np.random.uniform(0, 500, n)
        }
        return pl.DataFrame(data)