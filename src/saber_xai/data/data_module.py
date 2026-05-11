"""
Módulo de datos encargado de la ingesta de Parquet, división de splits,
y aplicación de Target Encoding sin data leakage.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder

from saber_xai.config import config

class ICFESDataset(Dataset):
    """Dataset de PyTorch para los datos del ICFES."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) # [batch_size, 1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DataModule:
    """
    Fábrica y administrador de datos.
    Carga el Parquet, maneja nulos, separa conjuntos y aplica transformaciones seguras.
    """
    def __init__(self):
        self.config = config
        self.encoder = TargetEncoder(target_type="continuous", random_state=config.random_state)
        
        # Datos en formato numpy/pandas
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

    def prepare_data(self) -> None:
        """
        Carga y procesa el dataset. Aplica Target Encoding al municipio,
        y previene el leakage ajustando el encoder exclusivamente en train.
        """
        try:
            df = pd.read_parquet(self.config.data_path)
        except Exception as e:
            # En caso de que el archivo no exista aún, generaremos un dummy para testing
            print(f"Advertencia: No se encontró {self.config.data_path}. Usando dummy data. Error: {e}")
            df = self._generate_dummy_data()

        # Separar X e y
        X = df.drop(columns=[self.config.target_col])
        y = df[self.config.target_col]

        # Divisiones (Train 70%, Val 15%, Test 15%)
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=0.15, random_state=self.config.random_state
        )
        # 0.15 / 0.85 = 0.1764 (aprox 15% del total para validación)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.1764, random_state=self.config.random_state 
        )

        # Manejo simple de nulos (imputación con mediana en numéricas)
        num_cols = self.X_train.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            median_val = self.X_train[col].median()
            self.X_train.fillna({col: median_val}, inplace=True)
            self.X_val.fillna({col: median_val}, inplace=True)
            self.X_test.fillna({col: median_val}, inplace=True)

        # Target Encoding
        # Aseguramos de que ajustamos (fit) SOLAMENTE en train para evitar leakage
        cat_col = self.config.cat_col_to_encode
        if cat_col in self.X_train.columns:
            # fit_transform en train
            self.X_train[[cat_col]] = self.encoder.fit_transform(
                self.X_train[[cat_col]], self.y_train
            )
            # Solo transform en val y test (Previene Leakage)
            self.X_val[[cat_col]] = self.encoder.transform(self.X_val[[cat_col]])
            self.X_test[[cat_col]] = self.encoder.transform(self.X_test[[cat_col]])

    def get_dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Retorna los DataLoaders de PyTorch para (Train, Val, Test)."""
        train_ds = ICFESDataset(self.X_train.values, self.y_train.values)
        val_ds = ICFESDataset(self.X_val.values, self.y_val.values)
        test_ds = ICFESDataset(self.X_test.values, self.y_test.values)

        train_loader = DataLoader(
            train_ds, batch_size=self.config.mlp_batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.config.mlp_batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_ds, batch_size=self.config.mlp_batch_size, shuffle=False
        )

        return train_loader, val_loader, test_loader

    def get_dmatrices(self) -> tuple[xgb.DMatrix, xgb.DMatrix, xgb.DMatrix]:
        """Retorna los DMatrix de XGBoost para (Train, Val, Test)."""
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        dval = xgb.DMatrix(self.X_val, label=self.y_val)
        dtest = xgb.DMatrix(self.X_test, label=self.y_test)
        
        return dtrain, dval, dtest

    def _generate_dummy_data(self) -> pd.DataFrame:
        """Genera datos simulados si el dataset real no existe (solo para desarrollo)."""
        np.random.seed(self.config.random_state)
        n_samples = 1000
        data = {
            self.config.cat_col_to_encode: np.random.choice(['BOGOTA', 'MEDELLIN', 'CALI'], n_samples),
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            self.config.target_col: np.random.uniform(0, 500, n_samples)
        }
        return pd.DataFrame(data)
