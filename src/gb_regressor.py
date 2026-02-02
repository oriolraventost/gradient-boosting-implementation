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
    """
    Gradient Boosting Regressor supporting subsampling and early stopping 
    on a validation set.
    """
    def __init__(self, weak_learner_key: str, dataset_config: dict):
        self.weak_learner_key: str = weak_learner_key
        self.dataset_config = dataset_config
        
        self.weights: list[float] = []
        self.weak_learners: list[object] = []
        
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
        init_params: dict = {},
        fit_params: dict = {}
    ) -> None:
        '''Executes flow.'''
        initial_constant = self._compute_initial_constant(y_train)
        self.weights.append(initial_constant)
        
        train_preds = np.full(len(y_train), initial_constant)
        valid_preds = np.full(len(y_valid), initial_constant)

        for iter in range(max_iter):            
            X_train_sub, y_train_sub, train_preds_sub = self._draw_subsample(X_train, y_train, train_preds, upsilon)
            pseudo_residuals_sub = self._compute_pseudo_residuals(y_train_sub, train_preds_sub)
            
            weak_learner = self._get_weak_learner(init_params)
            weak_learner.fit(X_train_sub, pseudo_residuals_sub, **fit_params)
            
            train_preds += learning_rate * weak_learner.predict(X_train)
            valid_preds += learning_rate * weak_learner.predict(X_valid)
            
            self.weights.append(learning_rate)
            self.weak_learners.append(weak_learner)
            
            if self._early_stopping_needed(y_valid, valid_preds, iter, patience):
                break
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        '''Generates predictions.'''
        logger.info(f"Generating predictions using {len(self.models)} weak learners...")

        preds = pd.Series(self.weights[0], index=X.index)

        for weight, weak_learner in zip(self.weights[1:], self.weak_learners):
            preds += weight * weak_learner.predict(X)

        return preds

    def save_model(self, file_path: str) -> None:
        '''Save weights and models.'''
        joblib.dump({"weights": self.weights, "weak_learners": self.weak_learners}, file_path)
        logger.info(f"Model saved to {file_path}")

    def load_model(self, file_path: str) -> None:
        '''Loads model.'''
        model_data = joblib.load(file_path)
        self.weights = model_data["weights"]
        self.weak_learners = model_data["weak_learners"]
    
    def _compute_initial_constant(self, y_train: pd.Series) -> float:
        """Initializes model with mean target value."""
        return y_train.mean()

    def _draw_subsample(self, X_train: pd.DataFrame, y_train: pd.Series, train_preds: np.ndarray, upsilon: float) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """Stochastic subsampling of training indices."""
        sample_size = int(upsilon * len(X_train))
        subsample_idx = np.random.choice(np.arange(len(X_train)), size=sample_size, replace=False)

        return X_train.iloc[subsample_idx], y_train.iloc[subsample_idx], train_preds[subsample_idx]

    def _compute_pseudo_residuals(self, y_train_sub: pd.Series, train_preds_sub: np.ndarray) -> pd.Series:
        """Computes residuals for the current subsample."""
        return pd.Series(
            - derivative_mean_squared_error(y_train_sub, train_preds_sub),
            index=y_train_sub.index
        )

    def _get_weak_learner(self, init_params: dict = {}) -> DecisionTreeRegressor | NNRegressor:
        if self.weak_learner_key == "neural_network":
            return NNRegressor(
                cat_cols=self.dataset_config["cat_cols"],
                num_cols=self.dataset_config["num_cols"],
                cat_cardinalities=self.dataset_config["cat_cardinalities"],
                **init_params
            )
        
        elif self.weak_learner_key == "decision_tree":
            return DecisionTreeRegressor(**init_params)

    def _early_stopping_needed(self, y_valid: pd.Series, valid_preds: np.ndarray, iter: int, patience: int) -> bool:
        """Calculates loss on Validation Set and handles early stopping."""
        current_loss = mean_squared_error(y_valid, valid_preds)
        logger.info(f"Iteration: {iter+1} | Validation loss: {current_loss:.4f}")

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_iter = iter
            logger.debug(f"New best validation loss: {current_loss:.4f} at iter {iter}")
            return False
        
        if iter - self.best_iter >= patience:
            logger.info(f"Early stopping triggered at iteration {iter}. Best iter: {self.best_iter}")
            valid_count = self.best_iter
            self.weights = self.weights[:valid_count + 1]
            self.weak_learners = self.weak_learners[:valid_count]

            self.best_iter = 0
            self.best_loss = float('inf')
            
            return True
            
        return False