import pandas as pd
import numpy as np
import xgboost as xgb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder, StandardScaler

# Asumiendo que 'config' es un objeto con tus rutas e hiperparámetros
from saber_xai.config import config

class ICFESDataset(Dataset):
    """Dataset de PyTorch para los datos del ICFES."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # Aseguramos que los datos entren como tensores de punto flotante de 32 bits
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) 

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DataModule:
    """
    Fábrica y administrador de datos para el proyecto SaberXAI.
    Centraliza la carga, división, imputación, codificación y escalado.
    """
    def __init__(self):
        self.config = config
        self.target_encoder = TargetEncoder(target_type="continuous", random_state=config.random_state)
        self.scaler = StandardScaler()
        
        # Contenedores para los sets procesados
        self.X_train, self.X_val, self.X_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None

    def prepare_data(self) -> None:
        """
        Flujo completo de ingeniería de datos protegiendo la integridad científica.
        """
        try:
            # Carga desde Parquet serializado
            df = pd.read_parquet(self.config.data_path)
        except Exception as e:
            print(f"⚠️ Error al cargar Parquet: {e}. Generando datos dummy...")
            df = self._generate_dummy_data()

        # 1. Separación de Target y Features
        X = df.drop(columns=[self.config.target_col])
        y = df[self.config.target_col]

        # 2. División de conjuntos (70% Train, 15% Val, 15% Test)
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=0.15, random_state=self.config.random_state
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.1764, random_state=self.config.random_state 
        )

        # 3. Imputación de valores nulos (Mediana)[cite: 1]
        # Se calcula en train y se aplica en todos para evitar sesgos
        num_cols = self.X_train.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            median_val = self.X_train[col].median()
            self.X_train[col] = self.X_train[col].fillna(median_val)
            self.X_val[col] = self.X_val[col].fillna(median_val)
            self.X_test[col] = self.X_test[col].fillna(median_val)

        # 4. Target Encoding (Específico para Municipio)[cite: 1]
        cat_to_encode = self.config.cat_col_to_encode # ej. 'cole_mcpio_ubicacion'
        if cat_to_encode in self.X_train.columns:
            # Fit solo en Train para prevenir data leakage[cite: 1]
            self.X_train[[cat_to_encode]] = self.target_encoder.fit_transform(
                self.X_train[[cat_to_encode]], self.y_train
            )
            self.X_val[[cat_to_encode]] = self.target_encoder.transform(self.X_val[[cat_to_encode]])
            self.X_test[[cat_to_encode]] = self.target_encoder.transform(self.X_test[[cat_to_encode]])

        # 5. One-Hot Encoding para categorías restantes (Evita el TypeError de PyTorch)
        # Identificamos columnas que aún son texto/objeto
        remaining_cats = self.X_train.select_dtypes(include=['object', 'category']).columns
        if len(remaining_cats) > 0:
            self.X_train = pd.get_dummies(self.X_train, columns=remaining_cats, drop_first=True)
            # Reindexamos Val y Test para asegurar que tengan las mismas columnas que Train
            self.X_val = pd.get_dummies(self.X_val, columns=remaining_cats, drop_first=True)
            self.X_val = self.X_val.reindex(columns=self.X_train.columns, fill_value=0)
            
            self.X_test = pd.get_dummies(self.X_test, columns=remaining_cats, drop_first=True)
            self.X_test = self.X_test.reindex(columns=self.X_train.columns, fill_value=0)

        # 6. Escalado Estándar (Z-Score)
        # Esencial para la convergencia de la red neuronal (MLP)
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)

        # 7. Asegurar tipo Float32 para PyTorch
        self.X_train = self.X_train.astype(np.float32)
        self.X_val = self.X_val.astype(np.float32)
        self.X_test = self.X_test.astype(np.float32)

    def get_dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Retorna objetos DataLoader listos para el entrenamiento de la MLP."""
        train_ds = ICFESDataset(self.X_train, self.y_train.values)
        val_ds = ICFESDataset(self.X_val, self.y_val.values)
        test_ds = ICFESDataset(self.X_test, self.y_test.values)

        return (
            DataLoader(train_ds, batch_size=self.config.mlp_batch_size, shuffle=True, num_workers=4, pin_memory=True),
            DataLoader(val_ds, batch_size=self.config.mlp_batch_size, shuffle=False, num_workers=4, pin_memory=True),
            DataLoader(test_ds, batch_size=self.config.mlp_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        )

    def get_dmatrices(self) -> tuple[xgb.DMatrix, xgb.DMatrix, xgb.DMatrix]:
        """Retorna objetos DMatrix optimizados para XGBoost."""
        return (
            xgb.DMatrix(self.X_train, label=self.y_train),
            xgb.DMatrix(self.X_val, label=self.y_val),
            xgb.DMatrix(self.X_test, label=self.y_test)
        )

    def _generate_dummy_data(self) -> pd.DataFrame:
        """Generador de datos sintéticos para pruebas de integración."""
        np.random.seed(self.config.random_state)
        n = 1000
        data = {
            self.config.cat_col_to_encode: np.random.choice(['MUN_A', 'MUN_B', 'MUN_C'], n),
            'cole_naturaleza': np.random.choice(['OFICIAL', 'NO OFICIAL'], n),
            'fami_estrato': np.random.randint(1, 7, n),
            'estu_edad': np.random.uniform(15, 60, n),
            self.config.target_col: np.random.uniform(0, 500, n)
        }
        return pd.DataFrame(data)