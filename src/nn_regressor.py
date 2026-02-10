import numpy as np
import pandas as pd
import logging
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset, random_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NNRegressor(nn.Module):
    """A PyTorch neural network regressor for tabular data.
    
    This model utilizes a hybrid architecture that processes numerical features 
    and categorical features separately before merging them. Categorical 
    features are passed through an Embedding layer, where the embedding 
    dimension is determined by the square root of the category cardinality.
    
    The backend is a Multi-Layer Perceptron (MLP) with Batch Normalization 
    and Dropout for regularization.

    Attributes:
        cat_cols (list[str]): Names of the categorical columns.
        num_cols (list[str]): Names of the numerical columns.
        embeddings (nn.ModuleList): List of embedding layers for each categorical feature.
        mlp (nn.Sequential): The core deep learning layers.
        device (torch.device): The hardware (CPU/GPU) where the model is loaded.
    """

    def __init__(self, cat_cols: list[str], num_cols: list[str], cat_cardinalities: list[int]):
        """Initializes the network layers and moves the model to the best available device.

        Args:
            cat_cols (list[str]): List of categorical feature names.
            num_cols (list[str]): List of numerical feature names.
            cat_cardinalities (list[int]): Number of unique values for each categorical feature.
        """
        super().__init__()
        
        self.cat_cols: list[str] = cat_cols
        self.num_cols: list[str] = num_cols
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(card + 1, int(np.ceil(np.sqrt(card + 1)))) 
            for card in cat_cardinalities
        ])
        
        input_size = len(num_cols) + sum(int(np.ceil(np.sqrt(card + 1))) for card in cat_cardinalities)
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 1)
        )
        
        self.device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.to(self.device)

    def _prepare_loaders(self, X: pd.DataFrame, y: pd.Series, batch_size: int = 2**10, train_split: float = 0.8):
        """Converts Pandas DataFrames into PyTorch DataLoaders.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.
            batch_size (int): Number of samples per gradient update.
            train_split (float): Ratio of data to use for internal training vs validation.

        Returns:
            tuple[DataLoader, DataLoader]: Training and validation data loaders.
        """
        X_nums_tensor = torch.tensor(X[self.num_cols].values, dtype=torch.float32)
        X_cats_tensor = torch.tensor(X[self.cat_cols].values, dtype=torch.long)
        y_tensor = torch.tensor(y.values.reshape(-1, 1), dtype=torch.float32)

        dataset = TensorDataset(X_nums_tensor, X_cats_tensor, y_tensor)

        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def forward(self, x_nums: torch.Tensor, x_cats: torch.Tensor) -> torch.Tensor:
        """Standard PyTorch forward pass.

        Args:
            x_nums (torch.Tensor): Tensor of numerical features.
            x_cats (torch.Tensor): Tensor of categorical features (integer encoded).

        Returns:
            torch.Tensor: Continuous output values.
        """
        emb_outputs = []
        for i, emb_layer in enumerate(self.embeddings):
            emb_outputs.append(emb_layer(x_cats[:, i]))
        
        x = torch.cat(emb_outputs + [x_nums], dim=1)
        
        return self.mlp(x)

    def fit(self, X: pd.DataFrame, y: pd.Series, learning_rate: float, weight_decay: float, epochs: int) -> None:
        """Trains the network using an AdamW optimizer and OneCycleLR scheduler.
        
        The training process includes internal validation splitting and restores the 
        weights from the epoch with the lowest validation loss (Early Stopping behavior).

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training labels.
            learning_rate (float): Maximum learning rate for the OneCycle policy.
            weight_decay (float): L2 regularization coefficient.
            epochs (int): Number of complete passes over the dataset.
        """
        train_loader, val_loader = self._prepare_loaders(X, y)
        criterion = nn.MSELoss()
        
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=learning_rate, 
            steps_per_epoch=len(train_loader), 
            epochs=epochs,
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        best_val_loss = float('inf')
        best_model_state = None
        
        logger.info(f"Starting training for {epochs} epochs on {self.device}.")

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for batch_X_nums, batch_X_cats, batch_y in train_loader:
                batch_X_nums = batch_X_nums.to(self.device)
                batch_X_cats = batch_X_cats.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                preds = self(batch_X_nums, batch_X_cats)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                
                scheduler.step()
                train_loss += loss.item()

            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for v_batch_X_nums, v_batch_X_cats, v_batch_y in val_loader:
                    v_batch_X_nums = v_batch_X_nums.to(self.device)
                    v_batch_X_cats = v_batch_X_cats.to(self.device)
                    v_batch_y = v_batch_y.to(self.device)
                    
                    v_preds = self(v_batch_X_nums, v_batch_X_cats)
                    v_loss = criterion(v_preds, v_batch_y)
                    val_loss += v_loss.item()
            
            avg_val = val_loss / len(val_loader)
            
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_model_state = copy.deepcopy(self.state_dict())
                
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch+1:02d} | Val MSE: {avg_val:.4f} | LR: {current_lr:.6f}")

        if best_model_state:
            self.load_state_dict(best_model_state)
            logger.info(f"Training complete. Best Val MSE: {best_val_loss:.4f} restored.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generates regression predictions for the given input data.

        Args:
            X (pd.DataFrame): Input feature matrix.

        Returns:
            np.ndarray: Flattened array of numerical predictions.
        """
        X_nums_tensor = torch.tensor(X[self.num_cols].values, dtype=torch.float32)
        X_cats_tensor = torch.tensor(X[self.cat_cols].values, dtype=torch.long)
        
        X_nums_tensor, X_cats_tensor = X_nums_tensor.to(self.device), X_cats_tensor.to(self.device)

        self.eval()
        with torch.no_grad():
            predictions = self.forward(X_nums_tensor, X_cats_tensor)
        
        return predictions.cpu().numpy().ravel()