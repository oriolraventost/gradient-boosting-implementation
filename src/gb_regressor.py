import logging
import joblib
import numpy as np
import pandas as pd

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
    """Gradient boosting regression supporting subsampling and early stopping.
    
    This model implements a stage-wise additive ensemble that minimizes Mean 
    Squared Error (MSE) by fitting subsequent weak learners to the negative
    gradient (pseudo-residuals) of the previous iterations. Scales target
    variable to mean 0 and std 1.

    Attributes:
        n_estimators (int): Maximum number of boosting stages.
        learning_rate (float): Step size shrinkage used in update.
        subsample (float): Fraction of observations to use for each
            weak learner.
        colsample_bytree (float): Fraction of features to use for 
            each weak learner.
        reg_lambda (float): L2 regularization term.
        early_stopping_rounds (int): Maximum number of non-improving
                iterations allowed.
        weak_learner_key (str): Weak learner to use (decision_tree or
            neural_network).
        weak_learner_config (dict): Hyperparameters of weak learner.
        initial_constant (float): Initial constant of boosting ensemble.
        weak_learners (list[tuple[DecisionTreeRegressor | NNRegressor,
            list]]): The collection of weak learners fitted during
            training and the features they were fitted on.
        best_mse (float): The minimum Mean Squared Error recorded on the 
            validation set.
        best_r2 (float): The R^2 recorded when Mean Squared Error was
            the lowest.
        best_iter (int): The iteration index that yielded the best_mse.
        start_timestamp (str | None): Fitting start timestamp.
        end_timestamp (str | None): Fitting end timestamp.
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
        """Initializes the model structure."""
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
        """Trains the boosting ensemble using stage-wise additive modeling.

        Args:
            X_train (np.ndarray): Training feature variables.
            y_train (np.ndarray): Training target variable.
            X_valid (np.ndarray): Validation features variables.
            y_valid (np.ndarray): Validation target variable.
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

            oob_idx = np.setdiff1d(np.arange(n_rows_train), subsample_idx)
            oob_preds = weak_learner.predict(X_train[oob_idx][:, colsample_bytree_idx])
            oob_mse = np.mean((pseudo_residuals[oob_idx] - oob_preds) ** 2)
            logger.info(f"OOB MSE: {oob_mse:.6f}")
            
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
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Aggregates predictions from the base learners.

        Args:
            X (np.ndarray): Feature variables to generate predictions for.

        Returns:
            np.ndarray: Regression predictions.
        """
        preds = np.full(len(X), self.initial_constant)

        for (weak_learner, colsample_bytree_idx) in self.weak_learners:
            update = self.learning_rate * weak_learner.predict(
                X[:, colsample_bytree_idx]
            )

            preds += update.reshape(preds.shape)

        return preds

    def save_model(self, file_path: str) -> None:
        """Saves the initial constant, the weak learners and the features
        they have been fitted on in a joblib file.

        Args:
            file_path (str): Destination path for the artifact.
        """
        model_data = {
            "initial_constant": self.initial_constant,
            "learning_rate": self.learning_rate,
            "weak_learners": self.weak_learners
            }
        
        joblib.dump(model_data, file_path)
        logger.info(f"Model saved to {file_path}")

    def load_model(self, file_path: str) -> None:
        """Loads a previously saved model from a joblib file.

        Args:
            file_path (str): Path to the saved model file.
        """
        model_data = joblib.load(file_path)
        logger.info(f"Model loaded from {file_path}")
        
        self.initial_constant = model_data["initial_constant"]
        self.learning_rate = model_data["learning_rate"]
        self.weak_learners = model_data["weak_learners"]
    
    def _compute_initial_constant(self, y_train: np.ndarray) -> float:
        """Calculates the optimal constant baseline (mean) for MSE loss.
        
        Args:
            y_train (np.ndarray): Training target variable.
        
        Returns:
            float: Constant value minimizing the loss function if taken
                as the prediction for all the observations (mean).
        """
        return y_train.mean()

    def _draw_subsample(
        self,
        n_rows_train: int,
        n_cols_train: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generates random indices for row and column sampling.
        
        This method implements the randomness needed to reduce overfitting
        by selecting a subset of observations and features for the current
        boosting iteration.

        Args:
            n_rows_train (int): Total number of rows available in
                the training set.
            n_cols_train (int): Total number of columns available in
                the training set.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - subsample_idx (np.ndarray): Array of integer indices
                    for row sampling.
                - colsample_bytree_idx (np.ndarray): Array of integer indices
                    for column sampling.
        """
        subsample_idx = np.random.choice(
            np.arange(n_rows_train),
            size=int(self.subsample * n_rows_train),
            replace=False
        )

        colsample_bytree_idx = np.random.choice(
            np.arange(n_cols_train),
            size=int(self.colsample_bytree * n_cols_train),
            replace=False
        )

        return subsample_idx, colsample_bytree_idx

    def _compute_pseudo_residuals(
        self,
        y_train_sub: np.ndarray,
        train_preds_sub: np.ndarray
    ) -> np.ndarray:
        """Computes the regularized Newton step for the current iteration.

        Calculates the step using a diagonal Hessian approximation (second-order
        derivative) with L2 regularization, matching the logic used in 
        extreme gradient boosting.
        
        Args:
            y_train_sub (np.ndarray): Subsampled training target variable.
            train_preds (np.ndarray): Subsampled rolling predictions on the
                training feature variables.
        
        Returns:
            np.ndarray: The regularized Newton update (-g / (h + lambda)), 
                representing the optimal step for the current iteration.
        """
        g = first_derivative_mean_squared_error(y_train_sub, train_preds_sub)
        h = second_derivative_mean_squared_error(y_train_sub, train_preds_sub)
        return - g / (h + np.full_like(train_preds_sub, self.reg_lambda))

    def _fit_weak_learner(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame
    ) -> DecisionTreeRegressor | NNRegressor:
        """Initializes and trains a weak learner.

        This internal method handles the factory logic for selecting the weak
        learner type and fitting it to the provided data.

        Args:
            X (pd.DataFrame): The features for the current boosting iteration.
            y (pd.DataFrame): The target (usually pseudo-residuals) to fit.

        Returns:
            DecisionTreeRegressor | NNRegressor: A trained weak learner
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
        """Monitors validation loss and rolls back weak learners if
        improvement stalls.

        Args:
            y_valid (np.ndarray): True validation targets.
            valid_preds (np.ndarray): Current predictions for the validation
                set.
            iter (int): Current iteration index.

        Returns:
            bool: True if training should terminate, False otherwise.
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