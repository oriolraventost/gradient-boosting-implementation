import logging
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import log_loss, accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.extmath import softmax

from src.utils import derivative_log_loss

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class GBClassifier:
    """Gradient boosting classification supporting subsampling and
    earlystopping.
    
    This model implements a stage-wise additive ensemble that minimizes 
    Cross-Entropy by fitting subsequent weak learners to the negative
    gradient (pseudo-residuals) of the previous iterations.

    Attributes:
        n_estimators (int): Maximum number of boosting stages.
        learning_rate (float): Step size shrinkage used in update.
        subsample (float): Fraction of observations to use for each
            weak learner.
        colsample_bytree (float): Fraction of features to use for 
            each weak learner.
        early_stopping_rounds (int): Maximum number of non-improving
                iterations allowed.
        max_depth (int): Maximum depth of trees.
        initial_constant (float): Initial constant of boosting ensemble.
        weak_learners (list[tuple(DecisionTreeRegressor, np.ndarray)]): The
            collection of weak learners fitted during training and the
            features they were fitted on.
        best_loss (float): The minimum Mean Squared Error recorded on the 
            validation set.
        best_iter (int): The iteration index that yielded the best_loss.
    """

    def __init__(
        self,
        n_estimators: int,
        learning_rate: float,
        subsample: float,
        colsample_bytree: float,
        early_stopping_rounds: int,
        max_depth: int
    ):
        """Initializes the gradient boosting classifier."""
        self.n_estimators: int = n_estimators
        self.learning_rate: float = learning_rate
        self.subsample: float = subsample
        self.colsample_bytree: float = colsample_bytree
        self.early_stopping_rounds: int = early_stopping_rounds
        self.max_depth: int = max_depth
        
        self.initial_constant: float | None = None
        self.weak_learners: list[tuple[DecisionTreeRegressor, list]] = []
        
        self.best_loss: float | None = None
        self.best_iter: int | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series
    ) -> None:
        """Trains the boosting ensemble using stage-wise additive modeling.
        
        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_valid (pd.DataFrame): Validation features for early stopping.
            y_valid (pd.Series): Validation labels for early stopping.
        """
        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()
        X_valid = X_valid.to_numpy()
        y_valid = y_valid.to_numpy()

        self.best_iter = 0
        self.best_loss = float('inf')

        n_rows_train, n_cols_train = X_train.shape
        
        self.initial_constant = self._compute_initial_constant(y_train)
        
        train_preds = np.full(
            (len(y_train), len(self.initial_constant)),
            self.initial_constant
        )

        valid_preds = np.full(
            (len(y_valid), len(self.initial_constant)),
            self.initial_constant
        )

        for iter in range(self.n_estimators):            
            subsample_idx, colsample_bytree_idx = self._draw_subsample(
                n_rows_train,
                n_cols_train
            )

            pseudo_residuals = self._compute_pseudo_residuals(
                y_train[subsample_idx],
                train_preds[subsample_idx]
            )
            
            weak_learner = DecisionTreeRegressor(
                max_depth=self.max_depth, 
                random_state=42
            )
            
            weak_learner.fit(
                X_train[subsample_idx][:, colsample_bytree_idx],
                pseudo_residuals
            )
            
            train_preds += self.learning_rate * weak_learner.predict(
                X_train[:, colsample_bytree_idx]
            )
            valid_preds += self.learning_rate * weak_learner.predict(
                X_valid[:, colsample_bytree_idx]
            )
            
            self.weak_learners.append((weak_learner, colsample_bytree_idx))
            
            if self._early_stopping_needed(y_valid, valid_preds, iter):
                break
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Aggregates predictions from the base learners.
        
        Args:
            X (pd.DataFrame): Features to generate predictions for.

        Returns:
            np.ndarray: Probability distribution over all the classes
                for each observation.
        """
        proba_preds = self.predict_proba(X)
        preds = np.argmax(proba_preds, axis=1)
        return preds
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Aggregates predictions from the base learners.
        
        Args:
            X (pd.DataFrame): Features to generate predictions for.

        Returns:
            np.ndarray: Probability distribution over all the classes
                for each observation.
        """
        X = X.to_numpy()

        proba_preds = np.full(
            (len(X), len(self.initial_constant)),
            self.initial_constant
        )

        for (weak_learner, colsample_bytree_idx) in self.weak_learners:
            proba_preds += self.learning_rate * weak_learner.predict(
                X[:, colsample_bytree_idx]
            )

        return softmax(proba_preds)

    def save_model(self, file_path: str) -> None:
        """Saves the initial constant, the weak learners and the features
        they have been fitted on in a joblib file.

        Args:
            file_path (str): Destination path for the artifact.
        """
        model_data = {
            "initial_constant": self.initial_constant,
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
        self.weak_learners = model_data["weak_learners"]
    
    def _compute_initial_constant(self, y_train: np.ndarray) -> np.ndarray:
        """Calculates the optimal constant baseline (logarithm of
        proportions) for CE loss.
        
        Args:
            y_train (np.ndarray): Training target variable.
        
        Returns:
            np.ndarray: Constant logits minimizing the loss function
                if taken as the prediction for all the observations.
        """
        _, counts = np.unique(y_train, return_counts=True)
        proportions = counts / len(y_train)
        return np.log(proportions)

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
        """Calculates the negative gradient of the loss function with respect
        to the predictions for the current iteration.
        
        Args:
            y_train_sub (np.ndarray): Subsampled training target variable.
            train_preds (np.ndarray): Subsampled rolling predictions on the
                training feature variables.
        
        Returns:
            np.ndarray: Negative gradient of the CE loss with respect to the
                predictions for the current iteration.
        """
        return - derivative_log_loss(y_train_sub, train_preds_sub)

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
            valid_preds (np.ndarray): Current logit predictions for the
                validation set.
            iter (int): Current iteration index.

        Returns:
            bool: True if training should terminate, False otherwise.
        """
        valid_probabilities = softmax(valid_preds)
        current_loss = log_loss(
            y_valid,
            valid_probabilities,
            labels=np.arange(valid_preds.shape[1])
        )

        predicted_classes = np.argmax(valid_probabilities, axis=1)
        current_accuracy = accuracy_score(y_valid, predicted_classes)

        if not (iter + 1) % 1:
            logger.info(
                f"Iteration: {iter+1} | "
                f"CE validation loss: {current_loss:.6f} | "
                f"Accuracy validation loss: {current_accuracy:.6f} | "

            )

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_iter = iter
            return False
        
        if iter - self.best_iter >= self.early_stopping_rounds:
            logger.info(
                f"Early stopping triggered at iteration {iter}. "
                f"Best iteration: {self.best_iter}"
            )
            
            valid_count = self.best_iter
            self.weights = self.weights[:valid_count + 1]
            self.weak_learners = self.weak_learners[:valid_count]
            
            return True
            
        return False