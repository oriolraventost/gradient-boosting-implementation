import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src.tree import XGBTree

class XGBRegressor:
    """
    Manager class for a Gradient Boosting ensemble using histogram-based decision trees.

    This class orchestrates the training of multiple `XGBTree` instances using second-order 
    gradient approximation. It implements key efficiency and regularization techniques 
    including quantile-based data discretization (binning), stochastic row and column 
    subsampling, and early stopping based on validation Mean Squared Error (MSE).

    Attributes:
        n_estimators (int): Maximum number of boosting iterations (trees).
        learning_rate (float): Step size shrinkage used in updates to prevent overfitting. 
            Also known as 'eta'.
        max_depth (int): Maximum depth of each individual tree.
        reg_lambda (float): L2 regularization term on weights (lambda).
        gamma (float): Minimum loss reduction required to make a further partition on a leaf node.
        subsample (float): Fraction of training instances to be randomly sampled for each tree.
        colsample_bytree (float): Fraction of features to be randomly sampled for each tree.
        early_stopping_rounds (int): Activates early stopping if validation loss does not 
            improve for a specified number of consecutive rounds.
        trees (list[XGBTree]): The collection of fitted histogram-based trees.
        bin_thresholds (list[np.ndarray]): Quantile boundaries used to discretize continuous 
            features, fitted during training.
        initial_constant (float | None): The starting prediction value (mean of the target 
            in the transformed space).
        target_mean (float | None): Mean of the target variable, used for Z-score normalization.
        target_std (float | None): Standard deviation of the target variable, used for 
            Z-score normalization.
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
        """
        Initializes the XGBRegressor with hyperparameter configurations.

        Args:
            n_estimators (int): Maximum number of trees to build.
            learning_rate (float): Shrinkage factor for each tree's contribution.
            max_depth (int): Depth limit for trees to control model complexity.
            reg_lambda (float): L2 regularization coefficient.
            gamma (float): Pruning parameter for tree growth.
            subsample (float): Row sampling ratio (0 to 1].
            colsample_bytree (float): Column sampling ratio per tree (0 to 1].
            early_stopping_rounds (int): Patience limit for validation loss improvement.
        """
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

        self.target_mean: float | None = None
        self.target_std: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        """
        Trains the ensemble using second-order gradients and monitors early stopping.

        The method normalizes the target variable, discretizes continuous features into 
        histograms for computational efficiency, and iteratively adds trees that fit 
        the gradients of the MSE loss function.

        Args:
            X (pd.DataFrame): Training feature matrix of shape (n_samples, n_features).
            y (pd.Series): Training target vector.
            X_val (pd.DataFrame): Validation feature matrix for early stopping.
            y_val (pd.Series): Validation target vector for early stopping.

        Returns:
            self: The fitted regressor instance.
        """
        self.target_mean = y.mean()
        self.target_std = y.std()

        y = (y - self.target_mean) / self.target_std
        y_val = (y_val - self.target_mean) / self.target_std

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

            print(f"iteration {i} | mse {loss}")
            
            if loss < best_loss:
                best_loss, wait = loss, 0
            else:
                wait += 1
            
            if wait >= self.early_stopping_rounds:
                print(f"Early stopping triggered at iteration {i}")
                break

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generates aggregated ensemble predictions for new input data.

        Applies the learned bin thresholds, computes the additive predictions 
        across all trees, and reverses the initial Z-score normalization.

        Args:
            X (pd.DataFrame): Input feature matrix.

        Returns:
            pd.Series: Vector of predictions indexed by the input DataFrame index.
        """
        X_arr = np.asarray(X)
        X_binned = self._apply_bin_mapping(X_arr)
        
        preds = np.full(len(X_arr), self.initial_constant)
        for tree in self.trees:
            preds += self.learning_rate * tree.predict(X_binned)

        preds = self.target_std * preds + self.target_mean
        return pd.Series(preds, index=X.index)
    
    def _get_bin_mapping(self, X: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Generates quantile-based thresholds and discretizes the feature matrix.
        
        Args:
            X (np.ndarray): Continuous feature matrix to discretize.

        Returns:
            tuple[np.ndarray, list[np.ndarray]]: 
                - X_binned: Integer-encoded matrix where each value represents a bin index.
                - bin_thresholds: List of arrays containing the quantile boundaries per feature.
        """
        bin_thresholds = []
        X_binned = np.zeros(X.shape, dtype=np.uint8)
        for j in range(X.shape[1]):
            t = np.unique(np.quantile(X[:, j], np.linspace(0, 1, 256)))
            bin_thresholds.append(t)
            X_binned[:, j] = np.clip(np.searchsorted(t, X[:, j], side='right') - 1, 0, len(t)-1)
        return X_binned, bin_thresholds

    def _apply_bin_mapping(self, X: np.ndarray) -> np.ndarray:
        """
        Discretizes a new feature matrix using the thresholds fitted on training data.
        
        Args:
            X (np.ndarray): Continuous feature matrix.

        Returns:
            np.ndarray: Matrix of discretized (binned) integer features.
        """
        X_binned = np.zeros(X.shape, dtype=np.uint8)
        for j in range(X.shape[1]):
            t = self.bin_thresholds[j]
            X_binned[:, j] = np.clip(np.searchsorted(t, X[:, j], side='right') - 1, 0, len(t)-1)
        return X_binned

if __name__=="__main__":
    processed_train_data = pd.read_csv("data/processed/insurance_train.csv")
    X_train, X_valid, y_train, y_valid = train_test_split(
            processed_train_data.drop(columns=["Premium Amount"]), 
            processed_train_data["Premium Amount"],
            test_size=0.2,
            random_state=42
        )
    
    xgb = XGBRegressor(
        n_estimators=100,
        learning_rate=0.01,
        max_depth=5,
        reg_lambda=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=3
    )
    xgb.fit(X_train, y_train, X_valid, y_valid)