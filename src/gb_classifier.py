import logging
import joblib
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import log_loss

from src.utils import derivative_log_loss

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GBClassifier:
    """Gradient boosting classification supporting subsampling and early stopping."""
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

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series
    ) -> None:
        '''Trains the boosting ensemble using stage-wise additive modeling.'''
        logger.info("Starting gradient boosting classification training...")
        
        self.initial_constant = self._compute_initial_constant(y_train)
        logger.info(f"Initial constant of the boosting ensemble (logarithm of proportions): {" ".join(self.initial_constant)}")
        
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
        '''Aggregates predictions from the base learners scaled by their weights.'''
        logger.info(f"Generating gradient boosting predictions...")

        preds = pd.Series(self.initial_constant, index=X.index)

        for decision_tree in self.learners:
            preds += self.learning_rate * decision_tree.predict(X)

        return preds

    def save_model(self, file_path: str) -> None:
        """Serializes the ensemble weights and decision trees to a file."""
        joblib.dump({"initial_constant": self.initial_constant, "learning_rate": self.learning_rate, "learners": self.learners}, file_path)
        logger.info(f"Model saved to {file_path}")

    def load_model(self, file_path: str) -> None:
        """Deserializes a previously saved model state from a file."""
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
        """Monitors validation loss and rolls back learners if improvement stalls."""
        current_loss = log_loss(y_valid, valid_preds)

        if not (iter + 1) % 10:
            logger.info(f"Iteration: {iter+1} | CE validation loss: {current_loss:.4f}")

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_iter = iter
            return False
        
        if iter - self.best_iter >= self.patience:
            logger.info(f"Early stopping triggered at iteration {iter}. Best iteration: {self.best_iter}")
            self.learners = self.learners[:self.best_iter]

            self.best_iter = 0
            self.best_loss = float('inf')
            
            return True
            
        return False