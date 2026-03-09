import numpy as np
import logging
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class NNRegressor(nn.Module):
    """A PyTorch neural network regressor for tabular data.

    Attributes:
        hidden_size (int): Number of units in each hidden layer.
        batch_size (int): Batch size for parallel processing.
        network (nn.Sequential): The core deep learning layers.
        device (torch.device): The hardware where the model is loaded.
    """

    def __init__(
        self,
        hidden_size: int,
        batch_size: int
    ):
        """Initializes the neural network regressor."""
        super().__init__()
        
        self.hidden_size: int = hidden_size
        self.batch_size: int = batch_size
        
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
        """Trains the network using an Adam optimizer and OneCycleLR
        scheduler. Includes gradient norm clipping.

        Args:
            X (np.ndarray): Features.
            y (np.ndarray): Labels.
        """
        input_size = X.shape[1]
        output_size = y.shape[1] if len(y.shape) > 1 else 1

        self._get_network(input_size, output_size)

        loader = self._prepare_loader(X, y)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters())

        self.train()
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            optimizer.zero_grad()
            
            preds = self(batch_X)
            loss = criterion(preds, batch_y.view_as(preds))
            
            loss.backward()
            optimizer.step()

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
        self.network = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.GELU(),

            nn.Linear(self.hidden_size, output_size)
        )

        self.network.to(self.device)

    def _prepare_loader(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> tuple[DataLoader, DataLoader]:
        """Converts NumPy arrays into PyTorch DataLoader.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target vector of shape (n_samples,).

        Returns:
            DataLoader: Shuffled DataLoader for training.
        """
        X_t = torch.from_numpy(X).to(torch.float32)
        y_t = torch.from_numpy(y).to(torch.float32)

        return DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=True
        )