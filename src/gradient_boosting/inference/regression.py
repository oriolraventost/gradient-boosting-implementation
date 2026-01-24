import joblib
import json
import logging
import argparse
import pandas as pd

from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GradientBoostingRegressionPredictor:
    """
    Loads a trained Gradient Boosting ensemble to make predictions on new data.
    """
    def __init__(self):
        self.dataset_name: str | None = None
        self.timestamp: str | None = None
        
        self.data: pd.DataFrame | None = None
        self.preds: pd.Series | None = None
        
        self.weights: list[float] | None = None
        self.models: list[object] | None = None

    def _load_data(self, dataset: str) -> None:
        '''Loads inference data.'''
        self.dataset_name = Path(dataset).stem.removesuffix("_test")
        path = Path(f"data/processed/{dataset}")
        logger.info(f"Loading inference data from {path}")
        self.data = pd.read_csv(path)

    def _resolve_model_path(self) -> Path:
        """Retrieves the latest timestamp for the dataset from the registry."""
        registry_path = Path("models/registry.json")
        try:
            registry = json.loads(registry_path.read_text(encoding='utf-8'))
            self.timestamp = registry.get(self.dataset_name).get("best")
        except (json.JSONDecodeError, FileNotFoundError):
            logger.error("Registry file not found or corrupted.")
            raise

        if not self.timestamp:
            raise ValueError(f"No model found in registry for dataset: {self.dataset_name}")
        
        return Path(f"models/{self.dataset_name}/{self.timestamp}.joblib")

    def _load_model(self) -> None:
        '''Loads best model.'''
        path = self._resolve_model_path()
        logger.info(f"Loading model from {path}")
        
        model_data = joblib.load(path)
        self.weights = model_data["weights"]
        self.models = model_data["models"]

    def _predict(self) -> None:
        '''Generates predictions.'''
        logger.info(f"Generating predictions using {len(self.models)} weak learners...")

        self.preds = pd.Series(self.weights[0], index=self.data.index, name="SalePrice")

        for weight, model in zip(self.weights[1:], self.models):
            self.preds += weight * model.predict(self.data)

    def _save_preds(self) -> None:
        '''Saves predictions'''
        save_dir = Path("data/predictions")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = save_dir / f"{self.dataset_name}_preds.csv"
        self.preds.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")

    def run(self, dataset: str) -> None:
        """
        Executes the prediction pipeline.
        
        Args:
            dataset: Filename of the input CSV in data/processed/.
            output_name: Optional filename for the output predictions. 
                         Defaults to 'preds_{dataset}'.
        """
        self._load_data(dataset)
        self._load_model()
        self._predict()
        self._save_preds()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Run training on a regression task")

    parser.add_argument("--dataset", type=str, required=True)

    args = parser.parse_args()

    gradient_boosting_regression_predictor = GradientBoostingRegressionPredictor()
    gradient_boosting_regression_predictor.run(
        dataset=args.dataset
    )