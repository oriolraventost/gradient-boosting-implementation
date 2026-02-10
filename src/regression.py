import numpy as np
import pandas as pd

from src.tree import XGBTree

class XGBRegressor:
    """Manager class for the Gradient Boosting ensemble using histogram-based trees.
    
    This class orchestrates the training of multiple XGBTree instances. It handles 
    data discretization (binning), stochastic row and column subsampling, 
    and early stopping based on validation MSE.

    Attributes:
        n_estimators (int): Maximum number of trees in the ensemble.
        learning_rate (float): Step size shrinkage used to prevent overfitting.
        trees (list[XGBTree]): The collection of fitted histogram-based trees.
        bin_thresholds (list[np.ndarray]): The quantile boundaries used to discretize 
            continuous features during training and inference.
        initial_constant (float | None): The global mean of the target, used as 
            the starting prediction.
    """

    def __init__(
        self,
        n_estimators: int,
        learning_rate: float,
        max_depth: int, 
        reg_lambda: float,
        gamma: float,
        subsample: float, 
        colsample_bytree: float,
        early_stopping_rounds: int
    ):
        """Initializes the ensemble with hyperparameter configurations."""
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
    
        self.trees: list[XGBTree] = []
        self.bin_thresholds: list[np.ndarray] = []
        self.initial_constant: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        """Trains the ensemble using second-order gradients and early stopping.
        
        The method converts input data to binned integers, initializes predictions 
        to the target mean, and iteratively adds trees to minimize MSE.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target values.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation target values.
        """
        X_arr, y_arr = np.asarray(X), np.asarray(y)
        X_val_arr, y_val_arr = np.asarray(X_val), np.asarray(y_val)
        
        X_binned, self.bin_thresholds = self._get_bin_mapping(X_arr)
        X_val_binned = self._apply_bin_mapping(X_val_arr)
        
        self.initial_constant = float(np.mean(y_arr))
        
        train_preds = np.full(len(y_arr), self.initial_constant)
        valid_preds = np.full(len(y_val_arr), self.initial_constant)
        
        best_loss, wait = float('inf'), 0
        n_rows, n_cols = X.shape[0], X.shape[1]

        for i in range(self.n_estimators):
            row_idx = np.random.choice(n_rows, int(self.subsample * n_rows), replace=False)
            col_idx = np.random.choice(n_cols, int(self.colsample_bytree * n_cols), replace=False)
            
            g = 2.0 * (train_preds[row_idx] - y_arr[row_idx])
            h = np.full(len(row_idx), 2.0)

            tree = XGBTree(self.max_depth, self.reg_lambda, self.gamma)
            tree.fit(X_binned[row_idx][:, col_idx], g, h, col_idx)
            
            self.trees.append(tree)
            
            train_preds += self.learning_rate * tree.predict(X_binned)
            valid_preds += self.learning_rate * tree.predict(X_val_binned)
            
            loss = np.mean((valid_preds - y_val_arr)**2)
            
            if loss < best_loss:
                best_loss, wait = loss, 0
            else:
                wait += 1
            
            if wait >= self.early_stopping_rounds:
                break

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generates predictions for new data.

        Args:
            X (pd.DataFrame): Feature matrix.

        Returns:
            pd.Series: Aggregated ensemble predictions.
        """
        X_arr = np.asarray(X)
        X_binned = self._apply_bin_mapping(X_arr)
        
        preds = np.full(len(X_arr), self.initial_constant)
        for tree in self.trees:
            preds += self.learning_rate * tree.predict(X_binned)
        return pd.Series(preds, index=X.index)
    
    def _get_bin_mapping(self, X: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Creates quantile-based bins for each feature in the training set.
        
        Args:
            X (np.ndarray): Continuous feature matrix.

        Returns:
            tuple[np.ndarray, list[np.ndarray]]: Binned matrix and the list of 
                thresholds for each feature.
        """
        bin_thresholds = []
        X_binned = np.zeros(X.shape, dtype=np.uint8)
        for j in range(X.shape[1]):
            t = np.unique(np.quantile(X[:, j], np.linspace(0, 1, 256)))
            bin_thresholds.append(t)
            X_binned[:, j] = np.clip(np.searchsorted(t, X[:, j], side='right') - 1, 0, len(t)-1)
        return X_binned, bin_thresholds

    def _apply_bin_mapping(self, X: np.ndarray) -> np.ndarray:
        """Applies previously fitted bin thresholds to new data.
        
        This ensures that test/validation data is discretized exactly like 
        the training data, preventing feature misalignment.
        """
        X_binned = np.zeros(X.shape, dtype=np.uint8)
        for j in range(X.shape[1]):
            t = self.bin_thresholds[j]
            X_binned[:, j] = np.clip(np.searchsorted(t, X[:, j], side='right') - 1, 0, len(t)-1)
        return X_binned