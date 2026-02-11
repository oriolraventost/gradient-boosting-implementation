import logging
import joblib
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

from src.nn_regressor import NNRegressor
from src.utils import derivative_mean_squared_error

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GBRegressor:
    """Gradient boosting regression supporting subsampling and early stopping.
    
    This model implements a stage-wise additive ensemble that minimizes Mean 
    Squared Error (MSE) by fitting subsequent trees to the negative gradient 
    (pseudo-residuals) of the previous iterations. Scales target variable to
    mean 0 and std 1.

    Attributes:
        max_iter (int): Maximum number of boosting stages (trees).
        learning_rate (float): Step size shrinkage used in update to prevent overfitting.
        patience (int): Number of iterations to wait for improvement before stopping.
        upsilon (float): Fraction of training data to use for each tree (0.0 to 1.0].
        max_depth (int): Maximum depth of weak learner trees.
        weights (list[float]): A list containing the initial constant (target mean) 
            followed by the learning rate used for each subsequent tree.
        weak_learners (list[DecisionTreeRegressor]): The collection of base 
            learners (weak regressors) fitted during training.
        best_loss (float): The minimum Mean Squared Error recorded on the 
            validation set.
        best_iter (int): The iteration index that yielded the best_loss.
        target_mean (float): Mean of the target variable in the training data.
        target_std (float): STD of the target variable in the training data.
    """

    def __init__(self, max_iter: int, learning_rate: float, upsilon: float, patience: int, max_depth: int):
        """Initializes the model structure and tracking for early stopping."""
        self.max_iter: int = max_iter
        self.learning_rate: float = learning_rate
        self.patience: int = patience
        self.upsilon: float = upsilon
        self.max_depth: int = max_depth
        
        self.weights: list[float] = []
        self.weak_learners: list[DecisionTreeRegressor] = []
        
        self.best_loss: float = float('inf')
        self.best_iter: int = 0

        self.target_mean: float | None = None
        self.target_std: float | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series
    ) -> None:
        """Trains the boosting ensemble using stage-wise additive modeling.

        Args:
            X_train (pd.DataFrame): Training feature matrix.
            y_train (pd.Series): Training target vector.
            X_valid (pd.DataFrame): Validation feature matrix for early stopping.
            y_valid (pd.Series): Validation target vector for early stopping.
        """
        logger.info("Starting gradient boosting regression training...")
        
        self.target_mean = y_train.mean()
        self.target_std = y_train.std()

        y_train = (y_train - self.target_mean) / self.target_std
        y_valid = (y_valid - self.target_mean) / self.target_std

        initial_constant = self._compute_initial_constant(y_train)
        self.weights.append(initial_constant)
        logger.info(f"Initial constant of the boosting ensemble (target mean): {initial_constant:.4f}")
        
        train_preds = np.full(len(y_train), initial_constant)
        valid_preds = np.full(len(y_valid), initial_constant)

        for iter in range(self.max_iter):            
            X_train_sub, y_train_sub, train_preds_sub = self._draw_subsample(X_train, y_train, train_preds, self.upsilon)
            pseudo_residuals_sub = self._compute_pseudo_residuals(y_train_sub, train_preds_sub)
            
            weak_learner = DecisionTreeRegressor(max_depth=self.max_depth)
            weak_learner.fit(
                X=X_train_sub,
                y=pseudo_residuals_sub
            )
            
            train_preds += self.learning_rate * weak_learner.predict(X_train)
            valid_preds += self.learning_rate * weak_learner.predict(X_valid)
            
            self.weights.append(self.learning_rate)
            self.weak_learners.append(weak_learner)
            
            if self._early_stopping_needed(y_valid, valid_preds, iter, self.patience):
                break
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Aggregates predictions from the base learners scaled by their weights.
           Scales them back using mean and STD of the training data.

        Args:
            X (pd.DataFrame): Feature matrix to generate predictions for.

        Returns:
            pd.Series: Continuous regression predictions.
        """
        logger.info(f"Generating gradient boosting predictions...")

        preds = pd.Series(self.weights[0], index=X.index)

        for weight, weak_learner in zip(self.weights[1:], self.weak_learners):
            preds += weight * weak_learner.predict(X)

        scaled_preds = self.target_std * preds + self.target_mean

        return scaled_preds

    def save_model(self, file_path: str) -> None:
        """Serializes the ensemble weights and weak learners to a file.

        Args:
            file_path (str): Destination path for the joblib artifact.
        """
        joblib.dump({"weights": self.weights, "weak_learners": self.weak_learners}, file_path)
        logger.info(f"Model saved to {file_path}")

    def load_model(self, file_path: str) -> None:
        """Deserializes a previously saved model state from a file.

        Args:
            file_path (str): Path to the saved model file.
        """
        model_data = joblib.load(file_path)
        logger.info(f"Model loaded from {file_path}")
        
        self.weights = model_data["weights"]
        self.weak_learners = model_data["weak_learners"]
    
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
            self.weights = self.weights[:valid_count + 1]
            self.decision_trees = self.decision_trees[:valid_count]

            self.best_iter = 0
            self.best_loss = float('inf')
            
            return True
            
        return False