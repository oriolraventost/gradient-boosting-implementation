import numpy as np
import pandas as pd
import joblib
import os
import json
import argparse

from datetime import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from typing import Type

from src.gradient_boosting import WEAK_LEARNERS_MAP
from src.gradient_boosting.utils import derivative_mean_squared_error

class GradientBoostingRegressionTrainer:
    """
    A custom Gradient Boosting Regression Trainer implementation
    supporting subsampling and early stopping.
    """
    def __init__(self):
        """Initializes the Gradient Boosting Regressor with empty state containers."""
        self.dataset_name: str | None = None
        self.timestamp: str | None = None

        self.X: pd.DataFrame | None = None
        self.y: pd.Series | None = None
        self.pseudo_residuals: pd.Series | None = None
        
        self.weights: list[float] = []
        self.models: list[DecisionTreeRegressor] = []
        
        self.preds: np.ndarray | None = None
        self.subsample_idx: np.ndarray | None = None
        
        # Stores the class definition (e.g., DecisionTreeRegressor), not an instance
        self.weak_learner_class: Type | None = None 
        
        self.best_test_loss: float = float('inf')
        self.best_iteration: int = 0

    def _load_data(self, dataset: str):
        self.dataset_name = os.path.splitext(dataset)[0]
        self.data = pd.read_csv(f"data/training/{dataset}")

    def _setup_data(self, target: str) -> None:
        """
        Prepares the feature matrix and target vector.
        
        Args:
            data: The full input DataFrame containing features and target.
            target: The name of the target column.
        """
        self.y = self.data[target].copy()
        self.X = self.data.drop(columns=[target])

    def _setup_weak_learner(self, weak_learner_name: str) -> None:
        """
        Selects the weak learner class from the global map.
        
        Args:
            weak_learner_name: Key to look up in WEAK_LEARNERS_MAP.
        """
        self.weak_learner_class = WEAK_LEARNERS_MAP[weak_learner_name]

    def _initialize_model(self) -> None:
        """
        Initializes the model with the mean of the target values (constant prediction).
        """
        y_mean = self.y.mean()
        self.weights.append(y_mean)
        self.preds = np.full(len(self.y), y_mean)
        
        self.best_test_loss = float('inf')
        self.best_iteration = 0

    def _draw_subsample(self, upsilon: float) -> None:
        """
        Selects a random subset of indices for stochastic boosting.
        
        Args:
            upsilon: The fraction of data to subsample (0.0 < upsilon <= 1.0).
        """
        n = len(self.X)
        sample_size = int(upsilon * n)
        self.subsample_idx = np.random.choice(np.arange(n), size=sample_size, replace=False)

    def _compute_pseudo_residuals(self) -> None:
        """Computes the negative gradient (pseudo-residuals) for the current subsample."""
        y_sub = self.y.values[self.subsample_idx]
        preds_sub = self.preds[self.subsample_idx]
        
        residuals = derivative_mean_squared_error(y=y_sub, y_preds=preds_sub)
        self.pseudo_residuals = pd.Series(residuals, index=self.X.index[self.subsample_idx])

    def _fit_weak_learner(self, **kwargs) -> None:
        """Fits a new weak learner instance to the current pseudo-residuals."""
        X_sub = self.X.iloc[self.subsample_idx]
        y_residuals = self.pseudo_residuals
        
        model = self.weak_learner_class(**kwargs)
        model.fit(X_sub, y_residuals)
        
        self.models.append(model)
        self.iteration += 1

    def _update_model(self, learning_rate: float) -> None:
        """
        Updates global predictions by adding the weighted predictions of the new tree.
        
        Args:
            learning_rate: The shrinkage factor (eta).
        """
        latest_model = self.models[-1]
        
        new_preds = latest_model.predict(self.X)
        self.preds += learning_rate * new_preds
        self.weights.append(learning_rate)

    def _check_early_stopping(self, iteration: int, patience: int) -> bool:
        """
        Checks if training should stop based on convergence.
        
        Returns:
            bool: True if training should stop, False otherwise.
        """
        current_loss = mean_squared_error(y_true=self.y, y_pred=self.preds)

        if current_loss < self.best_test_loss:
            self.best_test_loss = current_loss
            self.best_iteration = iteration
            return False
        
        if iteration - self.best_iteration >= patience:
            valid_count = self.best_iteration
            self.weights = self.weights[:valid_count + 1] 
            self.models = self.models[:valid_count]
            return True
            
        return False

    def _save_ensemble(self) -> None:
        """Saves the current model state to disk with a timestamped filename."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"models/{self.dataset_name}/{self.timestamp}.joblib"
        
        model_data = {
            "weights": self.weights,
            "models": self.models
        }
        joblib.dump(model_data, filename)

    def _update_registry(self):
        try:
            with open('data.json', 'r') as file:
                registry = json.load(file)
        except (json.JSONDecodeError, FileNotFoundError):
            registry = {}
        
        if self.dataset_name not in registry:
            registry[self.dataset_name] = self.timestamp
        
        with open("models/registry.json", 'w', encoding='utf-8') as file:
            json.dump(registry, file, indent=4)

    def run(
        self,
        dataset: str,
        target: str,
        weak_learner: str,
        upsilon: float,
        learning_rate: float,
        patience: int,
        max_iter: int,
        **model_params
    ) -> None:
        """
        Executes the Gradient Boosting training loop.

        Args:
            data: Input DataFrame.
            target: Target column name.
            weak_learner: Type of learner (e.g., 'decision_tree').
            upsilon: Subsample fraction.
            learning_rate: Step size.
            patience: Early stopping patience.
            M: Maximum number of iterations.
            **model_params: Hyperparameters for the weak learner.
        """
        self._load_data(dataset=dataset)
        self._setup_data(target=target)
        self._setup_weak_learner(weak_learner_name=weak_learner)
        self._initialize_model()

        for m in range(max_iter):
            print(f"Iteration: {m}")
            self._draw_subsample(upsilon=upsilon)
            self._compute_pseudo_residuals()
            self._fit_weak_learner(**model_params)
            self._update_model(learning_rate=learning_rate)
            
            if self._check_early_stopping(iteration=m, patience=patience):
                break
        
        self._save_ensemble()
        self._update_registry()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Run training on a regression task")

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--weak_learner", type=str, required=True)
    parser.add_argument("--upsilon", type=float, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--max_iter", type=int, required=True)

    args = parser.parse_args()

    gradient_boosting_regression_trainer = GradientBoostingRegressionTrainer()
    gradient_boosting_regression_trainer.run(
        dataset=args.dataset,
        target=args.target,
        weak_learner=args.weak_learner,
        upsilon=args.upsilon,
        learning_rate=args.learning_rate,
        patience=args.patience,
        max_iter=args.max_iter
    )