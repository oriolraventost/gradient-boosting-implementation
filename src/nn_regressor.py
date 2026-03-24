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
        epochs (int): Number of epochs for neural network training.
        learning_rate (float): Learning rate of gradient descent.
        hidden_size (list[int]): Number of units per hidden layer.
        batch_size (int): Batch size for training.
        network (nn.Sequential): Sequential container of the MLP layers.
        device (torch.device): Computing device used for model and data.
    """

    def __init__(
        self,
        epochs: int,
        learning_rate: float,
        hidden_size: list[int],
        batch_size: int
    ):
        """Initializes the neural network regressor."""
        super().__init__()
        
        self.epochs: int = epochs
        self.learning_rate: float = learning_rate
        self.hidden_size: list[int] = hidden_size
        self.batch_size: int = batch_size
        
        self.network: nn.Sequential | None = None

        self.device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the input through the feed-forward network.

        Args:
            x (torch.Tensor): Input feature tensor.

        Returns:
            torch.Tensor: Predicted continuous values.
        """     
        return self.network(x)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Trains the model using the provided NumPy datasets.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Target regression values.
        """
        input_size = X.shape[1]
        output_size = y.shape[1] if len(y.shape) > 1 else 1

        self._get_network(input_size, output_size)

        loader = self._prepare_loader(X, y)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), self.learning_rate)

        self.train()
        for _ in self.epochs:
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                
                preds = self(batch_X)
                loss = criterion(preds, batch_y.view_as(preds))
                
                loss.backward()
                optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates predictions for new input data.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Model predictions as a NumPy array.
        """
        X_t = torch.from_numpy(X).to(torch.float32)
        X_t = X_t.to(self.device)

        self.eval()
        with torch.no_grad():
            predictions = self.forward(X_t)
        
        return predictions.cpu().numpy()

    def _get_network(self, input_size: int, output_size: int) -> None:
        """Constructs the Multi-Layer Perceptron architecture.

        Args:
            input_size (int): Number of features in the input data.
            output_size (int): Dimension of the regression target.
        """
        self.network = nn.Sequential(
            nn.Linear(input_size, self.hidden_size[0]),
            nn.ReLU(),

            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),

            nn.Linear(self.hidden_size[1], output_size)
        )

        self.network.to(self.device)

    def _prepare_loader(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> tuple[DataLoader, DataLoader]:
        """Creates a PyTorch DataLoader from NumPy arrays.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target labels.

        Returns:
            DataLoader: Shuffled training data loader.
        """
        X_t = torch.from_numpy(X).to(torch.float32)
        y_t = torch.from_numpy(y).to(torch.float32)

        return DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=True
        )