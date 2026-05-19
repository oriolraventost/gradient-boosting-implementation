import logging
import joblib
import numpy as np

from datetime import datetime
from sklearn.metrics import log_loss, accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.extmath import softmax

from src.nn_regressor import NNRegressor
from src.cnn_regressor import CNNRegressor
from src.utils import (
    first_derivative_log_loss,
    second_derivative_log_loss
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class GBClassifier:
    """Gradient boosting classifier with subsampling and early stopping.

    This model implements a stage-wise additive ensemble minimizing
    Cross-Entropy loss by fitting weak learners to the regularized Newton 
    step of the log-loss function.

    Attributes:
        n_estimators (int): Maximum number of boosting stages.
        learning_rate (float): Step size shrinkage used in updates.
        subsample (float): Sample fraction used for each weak learner.
        colsample_bytree (float): Feature fraction used for each learner.
        reg_lambda (float): L2 regularization term for the Newton step.
        early_stopping_rounds (int): Limit without validation improvement.
        weak_learner_key (str): Chosen base learner identifier type.
        weak_learner_config (dict): Core hyperparameters for weak learners.
        initial_constant (np.ndarray | None): Center-log baseline logits.
        weak_learners (list | None): Tuple sequence of fitted models and columns.
        best_log_loss (float | None): Minimum recorded validation Log Loss.
        best_accuracy (float | None): Accuracy at the best Log Loss index.
        best_iter (int | None): Boosting iteration of the optimal model.
        start_timestamp (str | None): Formatted start time of fit execution.
        end_timestamp (str | None): Formatted termination time of fit execution.
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
        """Initializes ensemble hyperparameters and dynamic tracking states."""
        self.n_estimators: int = n_estimators
        self.learning_rate: float = learning_rate
        self.subsample: float = subsample
        self.colsample_bytree: float = colsample_bytree
        self.reg_lambda: float = reg_lambda
        self.early_stopping_rounds: int = early_stopping_rounds

        self.weak_learner_key: str = weak_learner_key
        self.weak_learner_config = weak_learner_config
        
        self.initial_constant: np.ndarray | None = None
        self.weak_learners: list[
            tuple[DecisionTreeRegressor | NNRegressor | CNNRegressor, list]
        ] | None = None
        
        self.best_log_loss: float | None = None
        self.best_accuracy: float | None = None
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
        """Trains the ensemble using stage-wise additive logit modeling.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training categorical targets.
            X_valid (np.ndarray): Validation evaluation features.
            y_valid (np.ndarray): Validation categorical targets.
        """
        start_time = datetime.now()
        self.start_timestamp = start_time.strftime("%Y_%m_%d_%H_%M")

        self.best_iter = 0
        self.best_log_loss = float('inf')
        self.best_accuracy = 0.0

        n_rows_train, n_cols_train, *extra = X_train.shape
        
        self.initial_constant = self._compute_initial_constant(y_train)
        self.weak_learners = []
        
        train_preds = np.full(
            (len(y_train), len(self.initial_constant)),
            self.initial_constant
        )

        valid_preds = np.full(
            (len(y_valid), len(self.initial_constant)),
            self.initial_constant
        )

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
        """Predicts the optimal target class label for each input sample.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Vector of predicted integer class boundaries.
        """
        proba_preds = self.predict_proba(X)
        return np.argmax(proba_preds, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts probability distributions via the ensemble softmax logits.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Probabilities map of shape (n_samples, n_classes).
        """
        proba_preds = np.full(
            (len(X), len(self.initial_constant)),
            self.initial_constant
        )

        for (weak_learner, colsample_bytree_idx) in self.weak_learners:
            update = self.learning_rate * weak_learner.predict(
                X[:, colsample_bytree_idx]
            )

            proba_preds += update.reshape(proba_preds.shape)

        return softmax(proba_preds)

    def save_model(self, file_path: str) -> None:
        """Saves core ensemble target definitions to a static joblib file.

        Args:
            file_path (str): Local filesystem route to write binary object.
        """
        model_data = {
            "initial_constant": self.initial_constant,
            "learning_rate": self.learning_rate,
            "weak_learners": self.weak_learners
            }
        
        joblib.dump(model_data, file_path)
        logger.info(f"Model saved to {file_path}")

    def load_model(self, file_path: str) -> None:
        """Restores explicit ensemble weights from a verified joblib source.

        Args:
            file_path (str): Target filesystem route holding saved weights.
        """
        model_data = joblib.load(file_path)
        logger.info(f"Model loaded from {file_path}")
        
        self.initial_constant = model_data["initial_constant"]
        self.learning_rate = model_data["learning_rate"]
        self.weak_learners = model_data["weak_learners"]
    
    def _compute_initial_constant(self, y_train: np.ndarray) -> np.ndarray:
        """Calculates centered log-proportions as initial model baseline.
        
        Args:
            y_train (np.ndarray): Unprocessed training array label keys.
        
        Returns:
            np.ndarray: Initial target baseline logit row array vector.
        """
        _, counts = np.unique(y_train, return_counts=True)
        proportions = counts / len(y_train)
        logits = np.log(proportions)
        return logits - np.mean(logits)

    def _draw_subsample(
        self,
        n_rows: int,
        n_cols: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generates matrix positions for stochastic row and feature splits.

        Args:
            n_rows (int): Aggregate height count dimensions from source data.
            n_cols (int): Aggregate width feature dimensions from source data.

        Returns:
            tuple[np.ndarray, np.ndarray]: Discrete sampled location indexes.
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
        """Computes the regularized Newton step for multiclass log-loss.

        Args:
            y_train_sub (np.ndarray): Subset targets matching batch scope.
            train_preds_sub (np.ndarray): Current raw tracking batch logits.

        Returns:
            np.ndarray: Numerical matrices defining specific negative updates.
        """
        g = first_derivative_log_loss(y_train_sub, train_preds_sub)
        h = second_derivative_log_loss(y_train_sub, train_preds_sub)
        return - g / (h + np.full_like(train_preds_sub, self.reg_lambda))
    
    def _fit_weak_learner(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> DecisionTreeRegressor | NNRegressor | CNNRegressor:
        """Instantiates and fits a single stage estimator targeting updates.

        Args:
            X (np.ndarray): Feature snapshot dimensions.
            y (np.ndarray): Computed directional residuals metrics target.

        Returns:
            DecisionTreeRegressor | NNRegressor | CNNRegressor: Fitted learner.
        """
        if self.weak_learner_key == "decision_tree":
            weak_learner = DecisionTreeRegressor(**self.weak_learner_config)
        
        elif self.weak_learner_key == "neural_network":
            weak_learner = NNRegressor(**self.weak_learner_config)
        
        elif self.weak_learner_key == "convolutional_neural_network":
            weak_learner = CNNRegressor(**self.weak_learner_config)
        
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
        """Evaluates convergence trends against validation history targets.

        Args:
            y_valid (np.ndarray): Multi-class validation ground truth entries.
            valid_preds (np.ndarray): Cumulative model predictive evaluations.
            iter (int): Current absolute indexing step position.

        Returns:
            bool: True if improvement stalls over threshold limits else False.
        """
        valid_probabilities = softmax(valid_preds)
        current_log_loss = log_loss(
            y_valid,
            valid_probabilities,
            labels=np.arange(valid_preds.shape[1])
        )

        predicted_classes = np.argmax(valid_probabilities, axis=1)
        current_accuracy = accuracy_score(y_valid, predicted_classes)

        logger.info(
            f"Iteration: {iter} | "
            f"Validation Log Loss: {current_log_loss:.6f} | "
            f"Validation Accuracy: {current_accuracy:.6f}"
        )

        if current_log_loss < self.best_log_loss:
            self.best_log_loss = current_log_loss
            self.best_accuracy = current_accuracy
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
