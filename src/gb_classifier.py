import logging
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.extmath import softmax

from src.utils import derivative_log_loss

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GBClassifier:
    """Gradient boosting classification supporting subsampling and early stopping.
    
    This ensemble model implements a stage-wise additive approach to minimize 
    Cross-Entropy loss. It supports stochastic gradient boosting through 
    row-wise subsampling (upsilon) and prevents overfitting via validation-based 
    early stopping.

    Attributes:
        max_iter (int): Maximum number of boosting stages (trees).
        learning_rate (float): Step size shrinkage used in update to prevent overfitting.
        patience (int): Number of iterations to wait for improvement before stopping.
        upsilon (float): Fraction of training data to use for each tree (0.0 to 1.0].
        max_depth (int): Maximum depth of the individual decision tree regressors.
        learners (list[DecisionTreeRegressor]): The collection of fitted trees.
        best_loss (float): The lowest validation loss observed during training.
    """

    def __init__(self, max_iter: int, learning_rate: float, patience: int, upsilon: float, max_depth: int):
        """Initializes the model structure and tracking for early stopping."""
        self.max_iter: int = max_iter
        self.learning_rate: float = learning_rate
        self.patience: int = patience
        self.upsilon: float = upsilon
        self.max_depth: int = max_depth
        
        self.initial_constant: np.ndarray | None = None
        self.learners: list[DecisionTreeRegressor] = []
        
        self.best_loss: float = float('inf')
        self.best_iter: int = 0

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series) -> None:
        """Trains the boosting ensemble using stage-wise additive modeling.
        
        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_valid (pd.DataFrame): Validation features for early stopping.
            y_valid (pd.Series): Validation labels for early stopping.
        """
        logger.info("Starting gradient boosting classification training...")

        self.best_iter = 0
        self.best_loss = float('inf')
        
        self.initial_constant = self._compute_initial_constant(y_train)
        
        train_preds = pd.DataFrame(np.tile(self.initial_constant, (len(y_train), 1)), columns=range(len(self.initial_constant)))
        valid_preds = pd.DataFrame(np.tile(self.initial_constant, (len(y_valid), 1)), columns=range(len(self.initial_constant)))

        for iter in range(self.max_iter):            
            X_train_sub, y_train_sub, train_preds_sub = self._draw_subsample(X_train, y_train, train_preds)
            pseudo_residuals_sub = self._compute_pseudo_residuals(y_train_sub, train_preds_sub)
            
            decision_tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
            decision_tree.fit(X_train_sub, pseudo_residuals_sub)
            
            train_preds += self.learning_rate * decision_tree.predict(X_train)
            valid_preds += self.learning_rate * decision_tree.predict(X_valid)
            
            self.learners.append(decision_tree)
            
            if self._early_stopping_needed(y_valid, valid_preds, iter):
                break
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Aggregates predictions from the base learners scaled by their weights.
        
        Args:
            X (pd.DataFrame): Features to generate predictions for.

        Returns:
            pd.Series: Raw model scores (logits) for each class.
        """
        logger.info(f"Generating gradient boosting predictions...")

        preds = pd.Series(self.initial_constant, index=X.index)

        for decision_tree in self.learners:
            preds += self.learning_rate * decision_tree.predict(X)

        return preds

    def save_model(self, file_path: str) -> None:
        """Serializes the ensemble weights and decision trees to a file.
        
        Args:
            file_path (str): Path to the destination file (usually .joblib).
        """
        joblib.dump({"initial_constant": self.initial_constant, "learning_rate": self.learning_rate, "learners": self.learners}, file_path)
        logger.info(f"Model saved to {file_path}")

    def load_model(self, file_path: str) -> None:
        """Deserializes a previously saved model state from a file.
        
        Args:
            file_path (str): Path to the saved model file.
        """
        model_data = joblib.load(file_path)
        logger.info(f"Model loaded from {file_path}")
        
        self.initial_constant = model_data["initial_constant"]
        self.learning_rate = model_data["learning_rate"]
        self.learners = model_data["learners"]
    
    def _compute_initial_constant(self, y_train: pd.Series) -> np.ndarray:
        """Calculates the optimal constant baseline (logarithm of proportions) for CE loss."""
        proportions = y_train.value_counts(normalize=True).sort_index().values
        return np.log(proportions)

    def _draw_subsample(self, X_train: pd.DataFrame, y_train: pd.Series, train_preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Performs row-wise sampling to introduce randomness into the fitting process."""
        sample_size = int(self.upsilon * len(X_train))
        subsample_idx = np.random.choice(np.arange(len(X_train)), size=sample_size, replace=False)

        return X_train.iloc[subsample_idx], y_train.iloc[subsample_idx], train_preds.iloc[subsample_idx]

    def _compute_pseudo_residuals(self, y_train_sub: pd.Series, train_preds_sub: pd.DataFrame) -> pd.DataFrame:
        """Calculates the negative gradient of the loss function for the current stage."""
        return - derivative_log_loss(y_train=y_train_sub, train_preds=train_preds_sub)

    def _early_stopping_needed(self, y_valid: pd.Series, valid_preds: pd.DataFrame, iter: int) -> bool:
        """Monitors validation loss and rolls back learners if improvement stalls.
        
        Args:
            y_valid (pd.Series): True labels for validation.
            valid_preds (pd.DataFrame): Current model predictions for validation.
            iter (int): The current iteration index.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        valid_probabilities = softmax(valid_preds.values)
        current_loss = log_loss(y_valid, valid_probabilities, labels=range(valid_preds.shape[1]))

        if not (iter + 1) % 10:
            logger.info(f"Iteration: {iter+1} | CE validation loss: {current_loss:.6f}")

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_iter = iter
            return False
        
        if iter - self.best_iter >= self.patience:
            logger.info(f"Early stopping triggered at iteration {iter}. Best iteration: {self.best_iter}")
            self.learners = self.learners[:self.best_iter]
            return True
            
        return False