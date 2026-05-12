"""
Modelo Multi-Layer Perceptron en PyTorch.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from saber_xai.config import config


class MLP(nn.Module):
    """Arquitectura modular MLP configurable desde el config."""
    def __init__(self, input_dim: int):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim

        for hidden_dim in config.mlp_hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.2))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))  # Output predictivo del puntaje (regresión)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MLPTrainer:
    """Clase para entrenar el MLP con Early Stopping y soporte para GPU."""

    def __init__(self, model: nn.Module, device: str = None):
        self.config = config
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.mlp_lr)

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Ciclo de entrenamiento por épocas con Early Stopping y CosineAnnealingLR.

        CosineAnnealingLR decae suavemente el lr desde el valor inicial hasta
        cerca de cero a lo largo de todas las épocas, sin warmup. Es compatible
        con Adam porque no eleva el lr por encima del valor inicial.

        El loop de batches usa _iter_batches en lugar de iterar el DataLoader
        directamente: evita el overhead de default_collate apilando tensores
        GPU individuales, que era la causa principal de la lentitud (~50 s/época).
        """
        best_val_loss = float('inf')
        patience_counter = 0

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.mlp_epochs,
            eta_min=self.config.mlp_lr * 0.01,
        )

        print(f"Iniciando entrenamiento MLP en {self.device}")

        for epoch in range(self.config.mlp_epochs):
            # --- Entrenamiento ---
            self.model.train()
            train_loss = 0.0
            n_train = 0
            for X_batch, y_batch in self._iter_batches(train_loader.dataset, shuffle=True):
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

                batch_n = X_batch.size(0)
                train_loss += loss.item() * batch_n
                n_train    += batch_n

            train_loss /= n_train
            scheduler.step()

            # --- Validación ---
            self.model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for X_batch, y_batch in self._iter_batches(val_loader.dataset, shuffle=False):
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)
                    batch_n = X_batch.size(0)
                    val_loss += loss.item() * batch_n
                    n_val    += batch_n
            val_loss /= n_val

            print(f"Epoch {epoch+1:03d}/{self.config.mlp_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

            # --- Early Stopping ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), "best_mlp_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= self.config.mlp_patience:
                    print(f"Early stopping activado en la época {epoch+1}")
                    break

    def _iter_batches(self, dataset: Dataset, shuffle: bool):
        """Iterador de batches optimizado para tensores GPU-resident.

        En lugar de usar DataLoader (que llama a default_collate apilando N
        tensores individuales de GPU uno a uno por batch), hace shuffle y
        slicing directamente sobre el tensor completo con una sola operación
        de gather por batch (X[idx]). Esto elimina el overhead de
        Python/collate que causaba ~50 s/época con datos ya en VRAM.

        Args:
            dataset: Dataset cuyas propiedades .X y .y son tensores
                (en CPU o CUDA, el device se respeta automáticamente).
            shuffle: Si True, aplica permutación aleatoria en el device
                del tensor antes de iterar.

        Yields:
            Tuplas (X_batch, y_batch) como tensores en el mismo device
            que dataset.X.
        """
        X, y = dataset.X, dataset.y
        n = len(X)
        indices = (
            torch.randperm(n, device=X.device) if shuffle
            else torch.arange(n, device=X.device)
        )
        batch_size = self.config.mlp_batch_size
        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            yield X[idx], y[idx]
