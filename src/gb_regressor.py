import logging
import joblib
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

from src.utils import derivative_mean_squared_error

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GBRegressor:
    """Gradient boosting regression supporting subsampling and early stopping.
    
    This model implements a stage-wise additive ensemble that minimizes Mean 
    Squared Error (MSE) by fitting subsequent trees to the negative gradient 
    (pseudo-residuals) of the previous iterations.

    Attributes:
        weights (list[float]): A list containing the initial constant (target mean) 
            followed by the learning rate used for each subsequent tree.
        decision_trees (list[DecisionTreeRegressor]): The collection of base 
            learners (weak regressors) fitted during training.
        best_loss (float): The minimum Mean Squared Error recorded on the 
            validation set.
        best_iter (int): The iteration index that yielded the best_loss.
    """

    def __init__(self):
        """Initializes the model structure and tracking for early stopping."""
        self.weights: list[float] = []
        self.decision_trees: list[DecisionTreeRegressor] = []
        
        self.best_loss: float = float('inf')
        self.best_iter: int = 0

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        upsilon: float,
        learning_rate: float,
        patience: int,
        max_iter: int,
        max_depth: int
    ) -> None:
        """Trains the boosting ensemble using stage-wise additive modeling.

        Args:
            X_train (pd.DataFrame): Training feature matrix.
            y_train (pd.Series): Training target vector.
            X_valid (pd.DataFrame): Validation feature matrix for early stopping.
            y_valid (pd.Series): Validation target vector for early stopping.
            upsilon (float): Subsample ratio (fraction of rows) for stochastic boosting.
            learning_rate (float): Shrinkage factor applied to each tree's output.
            patience (int): Iterations to wait for improvement before stopping.
            max_iter (int): Maximum number of trees to build.
            max_depth (int): Maximum depth of each decision tree.
        """
        logger.info("Starting gradient boosting regression training...")
        
        initial_constant = self._compute_initial_constant(y_train)
        self.weights.append(initial_constant)
        logger.info(f"Initial constant of the boosting ensemble (target mean): {initial_constant:.4f}")
        
        train_preds = np.full(len(y_train), initial_constant)
        valid_preds = np.full(len(y_valid), initial_constant)

        for iter in range(max_iter):            
            X_train_sub, y_train_sub, train_preds_sub = self._draw_subsample(X_train, y_train, train_preds, upsilon)
            pseudo_residuals_sub = self._compute_pseudo_residuals(y_train_sub, train_preds_sub)
            
            neural_network = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
            neural_network.fit(X_train_sub, pseudo_residuals_sub)
            
            train_preds += learning_rate * decision_tree.predict(X_train)
            valid_preds += learning_rate * decision_tree.predict(X_valid)
            
            self.weights.append(learning_rate)
            self.decision_trees.append(decision_tree)
            
            if self._early_stopping_needed(y_valid, valid_preds, iter, patience):
                break
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Aggregates predictions from the base learners scaled by their weights.

        Args:
            X (pd.DataFrame): Feature matrix to generate predictions for.

        Returns:
            pd.Series: Continuous regression predictions.
        """
        logger.info(f"Generating gradient boosting predictions...")

        preds = pd.Series(self.weights[0], index=X.index)

        for weight, decision_tree in zip(self.weights[1:], self.decision_trees):
            preds += weight * decision_tree.predict(X)

        return preds

    def save_model(self, file_path: str) -> None:
        """Serializes the ensemble weights and decision trees to a file.

        Args:
            file_path (str): Destination path for the joblib artifact.
        """
        joblib.dump({"weights": self.weights, "decision_trees": self.decision_trees}, file_path)
        logger.info(f"Model saved to {file_path}")

    def load_model(self, file_path: str) -> None:
        """Deserializes a previously saved model state from a file.

        Args:
            file_path (str): Path to the saved model file.
        """
        model_data = joblib.load(file_path)
        logger.info(f"Model loaded from {file_path}")
        
        self.weights = model_data["weights"]
        self.decision_trees = model_data["decision_trees"]
    
    def _compute_initial_constant(self, y_train: pd.Series) -> float:
        """Calculates the optimal constant baseline (mean) for MSE loss."""
        return y_train.mean()

    def _draw_subsample(self, X_train: pd.DataFrame, y_train: pd.Series, train_preds: np.ndarray, upsilon: float) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """Performs row-wise sampling to introduce randomness into the fitting process."""
        sample_size = int(upsilon * len(X_train))
        subsample_idx = np.random.choice(np.arange(len(X_train)), size=sample_size, replace=False)

        return X_train.iloc[subsample_idx], y_train.iloc[subsample_idx], train_preds[subsample_idx]

    def _compute_pseudo_residuals(self, y_train_sub: pd.Series, train_preds_sub: np.ndarray) -> pd.Series:
        """Calculates the negative gradient of the loss function for the current stage.
        
        In the case of MSE, the negative gradient is simply the difference 
        between the actual labels and the current predictions.
        """
        return pd.Series(
            - derivative_mean_squared_error(y_train_sub, train_preds_sub),
            index=y_train_sub.index
        )

    def _early_stopping_needed(self, y_valid: pd.Series, valid_preds: np.ndarray, iter: int, patience: int) -> bool:
        """Monitors validation loss and rolls back learners if improvement stalls.

        Args:
            y_valid (pd.Series): True validation targets.
            valid_preds (np.ndarray): Current predictions for the validation set.
            iter (int): Current iteration index.
            patience (int): Maximum number of non-improving iterations allowed.

        Returns:
            bool: True if training should terminate, False otherwise.
        """
        current_loss = mean_squared_error(y_valid, valid_preds)

        if not (iter + 1) % 10:
            logger.info(f"Iteration: {iter+1} | MSE validation loss: {current_loss:.4f}")

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_iter = iter
            return False
        
        if iter - self.best_iter >= patience:
            logger.info(f"Early stopping triggered at iteration {iter}. Best iteration: {self.best_iter}")
            valid_count = self.best_iter
            # Rollback to the best state
            self.weights = self.weights[:valid_count + 1]
            self.decision_trees = self.decision_trees[:valid_count]

            self.best_iter = 0
            self.best_loss = float('inf')
            
            return True
            
        return False