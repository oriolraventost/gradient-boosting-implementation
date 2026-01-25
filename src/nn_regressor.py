import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import logging
import copy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NNRegressor(nn.Module):
    """
    A professional, lean-constructor PyTorch regressor.
    """
    def __init__(self, input_size: int):
        super().__init__()
        self.device: torch.device = self._get_device()
        self.network: nn.Sequential = self._build_network(input_size)

        self.best_model_state: dict = None
        
        self.to(self.device)

    def _get_device(self) -> torch.device:
        """Determines the best available hardware accelerator."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _build_network(self, input_size: int) -> nn.Sequential:
        """Defines the model architecture separately to keep __init__ clean."""
        return nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, 1)
        )

    def _prepare_loaders(self, X: pd.DataFrame, y: pd.Series, batch_size: int = 32, train_split: float = 0.8):
        '''Turn pandas data into torch loaders.'''
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values.reshape(-1, 1), dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)

        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard PyTorch forward pass."""
        return self.network(x)
    
    def fit(
        self, 
        X: pd.DataFrame,
        y: pd.Series,
        epochs: int, 
        learning_rate: float,
        patience: int = 10
    ) -> None:
        """
        Main training loop with validation and early stopping.
        Note: The heavy lifting remains here to keep the logic encapsulated.
        """
        train_loader, val_loader = self._prepare_loaders(X, y)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        logger.info(f"Training started on {self.device}")

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                preds = self(batch_X)
                loss = criterion(preds, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for v_batch_X, v_batch_y in val_loader:
                    v_batch_X, v_batch_y = v_batch_X.to(self.device), v_batch_y.to(self.device)
                    v_preds = self(v_batch_X)
                    v_loss = criterion(v_preds, v_batch_y)
                    val_loss += v_loss.item()
            
            avg_val = val_loss / len(val_loader)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                epochs_no_improve = 0
                self.best_model_state = copy.deepcopy(self.state_dict())
            else:
                epochs_no_improve += 1
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1:03d} | Val MSE: {avg_val:.4f}")

            if epochs_no_improve >= patience:
                logger.info("Early stopping triggered.")
                break

        if self.best_model_state:
            self.load_state_dict(self.best_model_state)
            logger.info("Best weights restored.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generates predictions for the given input data."""
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        X_tensor = X_tensor.to(self.device)

        self.eval()
        with torch.no_grad():
            predictions = self.forward(X_tensor)
        
        return predictions.cpu().numpy().ravel()