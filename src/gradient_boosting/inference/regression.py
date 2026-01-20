import pandas as pd
import joblib
import os
import json

from sklearn.tree import DecisionTreeRegressor

class GradientBoostingRegressionPredictor:
    
    def __init__(self):
        self.dataset_name: str | None = None
        self.timestamp: str | None = None

        self.data: pd.DataFrame | None = None
        self.preds: pd.Series | None = None
        
        self.weights: list[float] | None = None
        self.weak_learners: list[DecisionTreeRegressor] | None = None
        
    def _load_data(self, dataset: str):
        self.dataset_name = os.path.splitext(dataset)[0]
        self.data = pd.read_csv(f"data/inference/{dataset}")
    
    def _get_models_path(self):
        try:
            with open('data.json', 'r') as file:
                registry = json.load(file)
        except (json.JSONDecodeError, FileNotFoundError):
            registry = {}
        
        self.timestamp = registry[self.dataset_name]

    def _load_model(self):
        model_data = joblib.load(f"models/{self.dataset_name}/{self.timestamp}")
        self.weights = model_data["weights"]
        self.models = model_data["models"]
    
    def _predict(self):
        self.preds = pd.Series(self.weights[0], index=range(len(self.data)))
        for weight, weak_learner in self.weights[1:], self.weak_learners:
            self.preds += weight * weak_learner.predict(self.data)
    
    def _save_preds(self, dataset: str):
        self.preds.to_csv(f"data/predictions/{dataset}")

    def run(
        self,
        dataset: str,
    ):
        self._load_data(dataset=dataset)
        self._get_models_path()
        self._load_model()
        self._predict()
        self._save_preds()