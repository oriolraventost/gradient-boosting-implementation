import json
import logging
import joblib
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from typing import Type
from pathlib import Path

from src import PROCESSED_DATA_PATH, MODELS_REGISTRY_PATH
from src.nn_regressor import NNRegressor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GBRegressionTrainer:
    """
    Gradient Boosting Regressor supporting subsampling and early stopping 
    on a validation set.
    """
    def __init__(self):
        self.dataset_name: str | None = None
        self.timestamp: str | None = None
        
        self.X_train: pd.DataFrame | None = None
        self.X_val: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_val: pd.Series | None = None
        
        self.weights: list[float] = []
        self.models: list[object] = []
        self.weak_learner_class: Type | None = None
        
        self.train_preds: np.ndarray | None = None
        self.pseudo_residuals: pd.Series | None = None
        self.subsample_idx: np.ndarray | None = None
        
        self.best_val_loss: float = float('inf')
        self.best_iteration: int = 0

    def _load_data(self, dataset: str, target: str, test_size: float = 0.2) -> None:
        '''Load data and split into train/test.'''
        self.dataset_name = Path(dataset).stem.removesuffix("_train")
        path = Path(PROCESSED_DATA_PATH) / dataset
        logger.info(f"Loading data from {path}")
        data = pd.read_csv(path)
        X = data.drop(columns=[target])
        y = data[target]
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        logger.info(f"Data split: {len(self.X_train)} train, {len(self.X_val)} validation samples.")

    def _get_weak_learner(self, weak_learner_name: str) -> None:
        '''Setup weak learner class.'''
        if weak_learner_name == "decision_tree": 
            self.weak_learner_class = DecisionTreeRegressor
        
        elif weak_learner_name == "neural_network":
            self.weak_learner_class = NNRegressor
        
        else:
            logger.error("Weak learner can only be decision tree or neural network.")

    def _initialize_model(self) -> None:
        """Initializes model with mean target value."""
        y_mean = self.y_train.mean()
        self.weights.append(y_mean)
        self.train_preds = np.full(len(self.y_train), y_mean)
        
        self.best_val_loss = float('inf')
        self.best_iteration = 0
        logger.info(f"Model initialized with mean: {y_mean:.4f}")

    def _draw_subsample(self, upsilon: float) -> None:
        """Stochastic subsampling of training indices."""
        n = len(self.X_train)
        sample_size = int(upsilon * n)
        self.subsample_idx = np.random.choice(np.arange(n), size=sample_size, replace=False)

    @staticmethod
    def _derivative_mean_squared_error(y: np.array, y_preds: np.array) -> np.array:
        """Compute the gradient of the mean squared error loss with respect to predictions."""
        return 2 * (y_preds - y)

    def _compute_pseudo_residuals(self) -> None:
        """Computes residuals for the current subsample."""
        y_sub = self.y_train.values[self.subsample_idx]
        preds_sub = self.train_preds[self.subsample_idx]
        
        residuals = - self._derivative_mean_squared_error(y=y_sub, y_preds=preds_sub)
        self.pseudo_residuals = pd.Series(residuals, index=self.X_train.index[self.subsample_idx])

    def _fit_weak_learner(self, init_params: dict | None = None, fit_params: dict | None = None) -> None:
        '''Fits weak learner on pseudo-residuals.'''
        init_params = init_params or {}
        fit_params = fit_params or {}

        X_sub = self.X_train.iloc[self.subsample_idx]
        if self.weak_learner_class == NNRegressor:
            init_params["input_size"] = X_sub.shape[1]

        model = self.weak_learner_class(**init_params)
        model.fit(X_sub, self.pseudo_residuals, **fit_params)
        self.models.append(model)

    def _update_train_preds(self, learning_rate: float) -> None:
        """Updates training predictions for the next residual calculation."""
        new_preds = self.models[-1].predict(self.X_train)
        self.train_preds += learning_rate * new_preds
        self.weights.append(learning_rate)

    def _check_early_stopping(self, iteration: int, patience: int) -> bool:
        """Calculates loss on Validation Set and handles early stopping."""
        val_preds = np.full(len(self.y_val), self.weights[0])
        for weight, model in zip(self.weights[1:], self.models):
            val_preds += weight * model.predict(self.X_val)

        current_loss = mean_squared_error(y_true=self.y_val, y_pred=val_preds)
        logger.info(f"Iteration: {iteration} | Validation loss: {current_loss:.4f}")

        if current_loss < self.best_val_loss:
            self.best_val_loss = current_loss
            self.best_iteration = iteration
            logger.debug(f"New best validation loss: {current_loss:.4f} at iter {iteration}")
            return False
        
        if iteration - self.best_iteration >= patience:
            logger.info(f"Early stopping triggered at iteration {iteration}. Best iter: {self.best_iteration}")
            valid_count = self.best_iteration
            self.weights = self.weights[:valid_count + 1]
            self.models = self.models[:valid_count]
            return True
            
        return False

    def _save_ensemble(self) -> None:
        '''Save weights and models.'''
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(f"models/{self.dataset_name}")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        filename = save_dir / f"{self.timestamp}.joblib"
        joblib.dump({"weights": self.weights, "models": self.models}, filename)
        logger.info(f"Model saved to {filename}")

    def _update_registry(self):
        """Updates the registry if the current model outperforms the previous best."""
        registry = {}
        
        path = Path(MODELS_REGISTRY_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            try:
                registry = json.loads(path.read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                logger.warning("Registry corrupted; initializing new registry.")

        current_record = registry.get(self.dataset_name, {"validation_loss": float('inf')})

        if self.best_val_loss < current_record["validation_loss"]:
            logger.info(f"New best model found for {self.dataset_name}!")
            registry[self.dataset_name] = {
                "best": self.timestamp,
                "validation_loss": self.best_val_loss
            }
            path.write_text(json.dumps(registry, indent=4), encoding='utf-8')
        else:
            logger.info("Current run did not beat the existing best model.")

    def run(self, dataset: str, target: str, weak_learner: str, upsilon: float, learning_rate: float,
            patience: int, max_iter: int, init_params: dict | None, fit_params: dict | None) -> None:
        '''Executes flow.'''
        self._load_data(dataset, target)
        self._get_weak_learner(weak_learner)
        self._initialize_model()

        logger.info(f"Starting training: max_iter={max_iter}, lr={learning_rate}, patience={patience}")

        for m in range(max_iter):            
            self._draw_subsample(upsilon)
            self._compute_pseudo_residuals()
            self._fit_weak_learner(init_params, fit_params)
            self._update_train_preds(learning_rate)
            
            if self._check_early_stopping(m, patience):
                break
        
        self._save_ensemble()
        self._update_registry()
        logger.info("Training complete.")
