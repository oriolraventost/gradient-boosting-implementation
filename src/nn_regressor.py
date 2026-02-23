import numpy as np
import pandas as pd
import logging
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset, random_split

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class NNRegressor(nn.Module):
    """A PyTorch neural network regressor for tabular data.
    
    The backend is a Multi-Layer Perceptron (MLP) with Batch Normalization 
    and Dropout for regularization.

    Attributes:
        epochs (int): Number of iterations on the full training data.
        learning_rate (float): Learning rate for the gradient descent updates.
        weight_decay (float): Weight regularization parameter.
        dropout (float): Dropout probability.
        hidden_size (int): Number of units in each hidden layer.
        batch_size (int): Batch size for parallel processing.
        network (nn.Sequential): The core deep learning layers.
        device (torch.device): The hardware where the model is loaded.
    """

    def __init__(
        self,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        dropout: float,
        hidden_size: int,
        batch_size: int
    ):
        """Initializes the neural network regressor."""
        super().__init__()
        
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        
        self.network: nn.Sequential | None = None

        self.device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard PyTorch forward pass.

        Args:
            x (torch.Tensor): Tensor of features.

        Returns:
            torch.Tensor: Output values.
        """        
        return self.network(x)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Trains the network using an AdamW optimizer and OneCycleLR
        scheduler.
        
        The training process includes internal validation splitting and
        restores the weights from the epoch with the lowest validation loss.

        Args:
            X (np.ndarray): Features.
            y (np.ndarray): Labels.
        """
        input_size = X.shape[1]
        output_size = y.shape[1] if len(y.shape) > 1 else 1

        self._get_network(input_size, output_size)

        train_loader, val_loader = self._prepare_loaders(X, y)
        criterion = nn.MSELoss()

        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=self.learning_rate, 
            steps_per_epoch=len(train_loader), 
            epochs=self.epochs,
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        best_val_mse = float('inf')
        best_model_state = None
        
        logger.info(
            f"Starting training for {self.epochs} epochs on {self.device}."
        )

        for epoch in range(1, self.epochs + 1):
            self.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                preds = self(batch_X)
                loss = criterion(preds, batch_y.view_as(preds))
                loss.backward()
                optimizer.step()
                
                scheduler.step()
                train_loss += loss.item()

            self.eval()
            val_mse = 0.0
            with torch.no_grad():
                for v_batch_X, v_batch_y in val_loader:
                    v_batch_X = v_batch_X.to(self.device)
                    v_batch_y = v_batch_y.to(self.device)
                    
                    v_preds = self(v_batch_X)
                    v_mse = criterion(v_preds, v_batch_y.view_as(v_preds))
                    val_mse += v_mse.item()
            
            avg_val = val_mse / len(val_loader)
            
            if avg_val < best_val_mse:
                best_val_mse = avg_val
                best_model_state = copy.deepcopy(self.state_dict())
                
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch} | Val MSE: {avg_val:.4f} | "
                f"LR: {current_lr:.4f}"
            )

        if best_model_state:
            self.load_state_dict(best_model_state)
            logger.info(
                f"Training complete. "
                f"Best Val MSE: {best_val_mse:.4f} restored."
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates regression predictions for the given input data.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Flattened array of numerical predictions.
        """
        X_t = torch.from_numpy(X).to(torch.float32)
        X_t = X_t.to(self.device)

        self.eval()
        with torch.no_grad():
            predictions = self.forward(X_t)
        
        return predictions.cpu().numpy()

    def _get_network(self, input_size: int, output_size: int) -> None:
        """Defines the feed-forward neural network architecture.

        Constructs a multi-layer perceptron (MLP) with two hidden layers, 
        incorporating Batch Normalization and Dropout for regularization. 

        Args:
            input_size (int): The number of input features.
            output_size (int): The dimension of the target.
        """
        self.network: nn.Sequential = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(self.hidden_size, output_size)
        )

        self.network.to(self.device)

    def _prepare_loaders(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> tuple[DataLoader, DataLoader]:
        """Converts Pandas DataFrames into PyTorch DataLoaders.

        Args:
            X (np.ndarray): Features.
            y (np.ndarray): Labels.

        Returns:
            tuple[DataLoader, DataLoader]: Training and validation data
                loaders.
        """
        X_t = torch.from_numpy(X).to(torch.float32)
        y_t = torch.from_numpy(y).to(torch.float32)

        dataset = TensorDataset(X_t, y_t)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        return train_loader, val_loader