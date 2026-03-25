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

class CNNRegressor(nn.Module):
    """A PyTorch convolutional neural network regressor.

    Attributes:
        epochs (int): Number of epochs for neural network training.
        learning_rate (float): Learning rate of gradient descent.
        channels (list[int]): Number of output channels for conv layers.
        kernel_size (int): Size of the square convolution kernel.
        pool_size (int): Size of the max pooling window.
        hidden_size (int): Number of neurons in the fully connected layer.
        batch_size (int): Size of training batches.
        network (nn.Sequential): The sequential container of model layers.
        device (torch.device): Hardware device (CPU/CUDA) used for tensors.
    """

    def __init__(
        self,
        epochs: int,
        learning_rate: float,
        channels: list[int],
        kernel_size: int,
        pool_size: int,
        hidden_size: int,
        batch_size: int
    ):
        """Initializes the convolutional neural network regressor."""
        super().__init__()
        
        self.epochs: int = epochs
        self.learning_rate: float = learning_rate
        self.channels: list[int] = channels
        self.kernel_size: int = kernel_size
        self.pool_size: int = pool_size
        self.hidden_size: int = hidden_size
        self.batch_size: int = batch_size
        
        self.network: nn.Sequential | None = None

        self.device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Regression predictions.
        """      
        return self.network(x)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Trains the model using the provided features and labels.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Target regression values.
        """
        in_channels = X.shape[1]
        output_size = y.shape[1] if len(y.shape) > 1 else 1
        
        image_size = X.shape[-1]

        conv1_out = image_size - self.kernel_size + 1
        pool1_out = conv1_out // self.pool_size

        conv2_out = pool1_out - self.kernel_size + 1
        pool2_out = conv2_out // self.pool_size

        linear_input = self.channels[1] * pool2_out * pool2_out
        
        self._get_network(in_channels, linear_input, output_size)

        loader = self._prepare_loader(X, y)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), self.learning_rate)

        self.train()
        for _ in range(self.epochs):
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                
                preds = self(batch_X)
                loss = criterion(preds, batch_y.view_as(preds))
                
                loss.backward()
                optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates predictions for the input data.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Numerical predictions as a NumPy array.
        """
        X_t = torch.from_numpy(X).to(torch.float32)
        X_t = X_t.to(self.device)

        self.eval()
        with torch.no_grad():
            predictions = self.forward(X_t)
        
        return predictions.cpu().numpy()

    def _get_network(
        self,
        in_channels: int,
        linear_input: int,
        output_size: int
    ) -> None:
        """Builds the nn.Sequential network architecture.

        Args:
            in_channels (int): Number of input image channels.
            linear_input (int): Flattened size after conv/pool layers.
            output_size (int): Dimension of the target output.
        """
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, self.channels[0], self.kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(self.pool_size),

            nn.Conv2d(self.channels[0], self.channels[1], self.kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(self.pool_size),

            nn.Flatten(),
            
            nn.Linear(linear_input, self.hidden_size),
            nn.ReLU(),

            nn.Linear(self.hidden_size, output_size)
        )

        self.network.to(self.device)

    def _prepare_loader(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> tuple[DataLoader, DataLoader]:
        """Wraps NumPy arrays into a shuffled PyTorch DataLoader.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target labels.

        Returns:
            DataLoader: Prepared training data loader.
        """
        X_t = torch.from_numpy(X).to(torch.float32)
        y_t = torch.from_numpy(y).to(torch.float32)

        return DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=True
        )