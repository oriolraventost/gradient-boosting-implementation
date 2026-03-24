import logging
import joblib
import numpy as np

from datetime import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

from src.nn_regressor import NNRegressor
from src.utils import (
    first_derivative_mean_squared_error,
    second_derivative_mean_squared_error
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class GBRegressor:
    """Gradient boosting regressor with subsampling and early stopping.

    This model implements a stage-wise additive ensemble that minimizes MSE by
    fitting weak learners to the regularized Newton step of the loss function.

    Attributes:
        n_estimators (int): Maximum number of boosting stages.
        learning_rate (float): Step size shrinkage used in each update.
        subsample (float): Fraction of samples used for each weak learner.
        colsample_bytree (float): Fraction of features used for each learner.
        reg_lambda (float): L2 regularization term for the Newton step.
        early_stopping_rounds (int): Iterations allowed without MSE
            improvement.
        weak_learner_key (str): Type of learner.
        weak_learner_config (dict): Hyperparameters for the weak learners.
        initial_constant (float): The global mean of the target variable.
        weak_learners (list): Fitted learners and their associated feature
            indices.
        best_mse (float): Lowest recorded Mean Squared Error on validation
            set.
        best_r2 (float): R^2 score corresponding to the best_mse.
        best_iter (int): Iteration index that achieved best_mse.
        start_timestamp (str): Formatted start time of the fit process.
        end_timestamp (str): Formatted end time of the fit process.
    """

    def __init__(
        self,
        n_estimators: int,
        learning_rate: float,
        subsample: float,
        colsample_bytree: float,
        reg_lambda: float,
        early_stopping_rounds: int,
        weak_learner_key: str,
        weak_learner_config: dict
    ):
        """Initializes the gradient boosting regressor."""
        self.n_estimators: int = n_estimators
        self.learning_rate: float = learning_rate
        self.subsample: float = subsample
        self.colsample_bytree: float = colsample_bytree
        self.reg_lambda: float = reg_lambda
        self.early_stopping_rounds: int = early_stopping_rounds

        self.weak_learner_key: str = weak_learner_key
        self.weak_learner_config: dict = weak_learner_config
        
        self.initial_constant: float | None = None
        self.weak_learners: list[
            tuple[DecisionTreeRegressor | NNRegressor, list]
        ] = []
        
        self.best_mse: float | None = None
        self.best_r2: float | None = None
        self.best_iter: int | None = None

        self.start_timestamp: str | None = None
        self.end_timestamp: str | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray
    ) -> None:
        """Trains the ensemble using stage-wise additive modeling.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training targets.
            X_valid (np.ndarray): Validation features.
            y_valid (np.ndarray): Validation targets.
        """
        start_time = datetime.now()
        self.start_timestamp = start_time.strftime("%Y_%m_%d_%H_%M")

        self.best_iter = 0
        self.best_mse = float('inf')

        n_rows_train, n_cols_train = X_train.shape

        self.initial_constant = self._compute_initial_constant(y_train)
        
        train_preds = np.full(len(y_train), self.initial_constant)
        valid_preds = np.full(len(y_valid), self.initial_constant)

        for iter in range(1, self.n_estimators + 1):            
            pseudo_residuals = self._compute_pseudo_residuals(
                y_train,
                train_preds
            )

            subsample_idx, colsample_bytree_idx = self._draw_subsample(
                n_rows_train,
                n_cols_train
            )
            
            weak_learner = self._fit_weak_learner(
                X_train[subsample_idx][:, colsample_bytree_idx],
                pseudo_residuals[subsample_idx]
            )
            
            train_update = self.learning_rate * weak_learner.predict(
                X_train[:, colsample_bytree_idx]
            )

            train_preds += train_update.reshape(train_preds.shape)

            valid_update = self.learning_rate * weak_learner.predict(
                X_valid[:, colsample_bytree_idx]
            )
            
            valid_preds += valid_update.reshape(valid_preds.shape)
            
            self.weak_learners.append((weak_learner, colsample_bytree_idx))
            
            if self._early_stopping_needed(y_valid, valid_preds, iter):
                break
            
        end_time = datetime.now()
        self.end_timestamp = end_time.strftime("%Y_%m_%d_%H_%M")
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Aggregates predictions from all fitted weak learners.

        Args:
            X (np.ndarray): Input features for prediction.

        Returns:
            np.ndarray: Predicted regression values.
        """
        preds = np.full(len(X), self.initial_constant)

        for (weak_learner, colsample_bytree_idx) in self.weak_learners:
            update = self.learning_rate * weak_learner.predict(
                X[:, colsample_bytree_idx]
            )

            preds += update.reshape(preds.shape)

        return preds

    def save_model(self, file_path: str) -> None:
        """Saves the model state to a joblib artifact.

        Args:
            file_path (str): Destination path for the saved model.
        """
        model_data = {
            "initial_constant": self.initial_constant,
            "learning_rate": self.learning_rate,
            "weak_learners": self.weak_learners
            }
        
        joblib.dump(model_data, file_path)
        logger.info(f"Model saved to {file_path}")

    def load_model(self, file_path: str) -> None:
        """Loads a model state from a joblib artifact.

        Args:
            file_path (str): Path to the saved model file.
        """
        model_data = joblib.load(file_path)
        logger.info(f"Model loaded from {file_path}")
        
        self.initial_constant = model_data["initial_constant"]
        self.learning_rate = model_data["learning_rate"]
        self.weak_learners = model_data["weak_learners"]
    
    def _compute_initial_constant(self, y_train: np.ndarray) -> float:
        """Calculates the mean target value as the starting baseline.

        Args:
            y_train (np.ndarray): Training target variable.

        Returns:
            float: Mean of the target variable.
        """
        return y_train.mean()

    def _draw_subsample(
        self,
        n_rows: int,
        n_cols: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generates random row and column indices for stochastic boosting.

        Args:
            n_rows (int): Total rows in training set.
            n_cols (int): Total columns in training set.

        Returns:
            tuple[np.ndarray, np.ndarray]: Row and column indices.
        """
        subsample_idx = np.random.choice(
            np.arange(n_rows),
            size=int(self.subsample * n_rows),
            replace=False
        )

        colsample_bytree_idx = np.random.choice(
            np.arange(n_cols),
            size=int(self.colsample_bytree * n_cols),
            replace=False
        )

        return subsample_idx, colsample_bytree_idx

    def _compute_pseudo_residuals(
        self,
        y_train_sub: np.ndarray,
        train_preds_sub: np.ndarray
    ) -> np.ndarray:
        """Computes the regularized Newton step for the current iteration.

        Args:
            y_true (np.ndarray): Target values.
            y_pred (np.ndarray): Current ensemble predictions.

        Returns:
            np.ndarray: The regularized update step.
        """
        g = first_derivative_mean_squared_error(y_train_sub, train_preds_sub)
        h = second_derivative_mean_squared_error(y_train_sub, train_preds_sub)
        return - g / (h + np.full_like(train_preds_sub, self.reg_lambda))

    def _fit_weak_learner(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> DecisionTreeRegressor | NNRegressor:
        """Instantiates and fits a weak learner based on the configured key.

        Args:
            X (np.ndarray): Feature subset.
            y (np.ndarray): Target pseudo-residuals.

        Returns:
            DecisionTreeRegressor | NNRegressor: A fitted weak learner
                instance.
        """
        if self.weak_learner_key == "decision_tree":
            weak_learner = DecisionTreeRegressor(**self.weak_learner_config)
        
        elif self.weak_learner_key == "neural_network":
            weak_learner = NNRegressor(**self.weak_learner_config)
        
        else:
            raise ValueError(
                f"Invalid weak learner key: {self.weak_learner_key}"
            )

        weak_learner.fit(X, y)
        return weak_learner
    
    def _early_stopping_needed(
        self,
        y_valid: np.ndarray,
        valid_preds: np.ndarray,
        iter: int
    ) -> bool:
        """Checks if validation performance has stalled to stop training.

        Args:
            y_valid (np.ndarray): Ground truth validation targets.
            valid_preds (np.ndarray): Current validation predictions.
            iter_idx (int): The current boosting iteration.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        current_mse = mean_squared_error(y_valid, valid_preds)
        current_r2 = r2_score(y_valid, valid_preds)

        if not iter % 1:
            logger.info(
                f"Iteration: {iter} | "
                f"Validation MSE: {current_mse:.6f} | "
                f"Validation R^2: {current_r2:.6f} "
            )

        if current_mse < self.best_mse:
            self.best_mse = current_mse
            self.best_r2 = current_r2
            self.best_iter = iter
            return False
        
        if iter - self.best_iter >= self.early_stopping_rounds:
            logger.info(
                f"Early stopping triggered at iteration {iter}. "
                f"Best iteration: {self.best_iter}"
            )

            self.weak_learners = self.weak_learners[:self.best_iter]
            return True
            
        return False