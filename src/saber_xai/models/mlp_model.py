"""
Modelo Multi-Layer Perceptron en PyTorch.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from saber_xai.config import config
import os

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
            
        layers.append(nn.Linear(in_dim, 1)) # Output predictivo del puntaje (regresión)
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
        """Ciclo de entrenamiento por épocas con Early Stopping y OneCycleLR.
        
        OneCycleLR es especialmente útil con batches grandes: aplica warmup
        hasta max_lr y luego decae, compensando el mayor ruido de gradiente.
        Se inicializa aquí para conocer steps_per_epoch del loader real.
        """
        best_val_loss = float('inf')
        patience_counter = 0

        scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.mlp_lr,
            steps_per_epoch=len(train_loader),
            epochs=self.config.mlp_epochs,
            pct_start=0.3,       # 30% de los pasos en warmup
            anneal_strategy='cos',
        )
        
        print(f"Iniciando entrenamiento MLP en {self.device}")
        
        for epoch in range(self.config.mlp_epochs):
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                scheduler.step()  # OneCycleLR se actualiza por batch, no por época
                
                train_loss += loss.item() * X_batch.size(0)
                
            train_loss /= len(train_loader.dataset)
            
            # Validación
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)
                    val_loss += loss.item() * X_batch.size(0)
            val_loss /= len(val_loader.dataset)
            
            print(f"Epoch {epoch+1:03d}/{self.config.mlp_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            
            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), "best_mlp_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= self.config.mlp_patience:
                    print(f"Early stopping activado en la época {epoch+1}")
                    break
