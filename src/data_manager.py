import numpy as np
import pandas as pd
import json
import yaml

from pathlib import Path
from sklearn.preprocessing import RobustScaler, LabelEncoder

from src import *
from src.gb_regressor import GBRegressor
from src.gb_classifier import GBClassifier

class DataManager:
    """Handles the ingestion and persistence of files.

    Attributes:
        raw_dir (Path | None): Directory where raw dataset files are stored.
        processed_dir (Path | None): Directory where processed CSVs are saved.
        model_dir (Path  | None): Directory where trained models are saved.
        id_column (str | None): The column name used as the identifier.
        target (str | None): The column name for the target variable.
        problem_type (str | None): Regression or classification.
    """

    def __init__(self):
        """Initializes the DataManager attributes."""
        self.raw_dir: Path | None = None
        self.processed_dir: Path | None = None
        self.model_dir: Path | None = None

        self.id_column: str | None = None
        self.target: str | None = None
        self.problem_type: str | None = None

    def load_config(self) -> dict:
        """Loads the configuration from a YAML file.

        Returns:
            dict: A dictionary containing the configuration settings.
        """
        with open(DATASETS_CONFIG_PATH, "r") as f:
            datasets_config = yaml.safe_load(f)
        
        with open(MODELING_CONFIG_PATH, "r") as f:
            modeling_config = yaml.safe_load(f)
        
        active_dataset = modeling_config["main"]["active_dataset"]
        
        self.raw_dir = Path(DATA_PATH) / active_dataset / "raw"

        self.processed_dir = Path(DATA_PATH) / active_dataset / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_dir = Path(MODELS_PATH) / active_dataset
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.id_column = datasets_config[active_dataset]["id_column"]
        self.target = datasets_config[active_dataset]["target"]

        return datasets_config, modeling_config
    
    def load_raw_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Loads raw dataset CSV files as pandas DataFrames.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Raw training and test data.
        """
        train_path = self.raw_dir / "train.csv"
        test_path = self.raw_dir / "test.csv"

        train = pd.read_csv(train_path, index_col=self.id_column)
        test = pd.read_csv(test_path, index_col=self.id_column)
        
        return train, test
    
    def save_processed_data(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame
    ) -> None:
        """Persists processed DataFrames to the disk.

        Args:
            train (pd.DataFrame): The cleaned training data.
            test (pd.DataFrame): The cleaned test data.
        """
        train.to_csv(self.processed_dir / "train.csv")
        test.to_csv(self.processed_dir / "test.csv")

    def load_processed_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Loads previously saved processed data.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Processed training and test
                data.
        """
        train = pd.read_csv(
            self.processed_dir / "train.csv",
            index_col=self.id_column
        )

        test = pd.read_csv(
            self.processed_dir / "test.csv",
            index_col=self.id_column
        )

        return train, test
    
    def save_model(
        self,
        gradient_boosting_config: dict,
        weak_learner_config: dict,
        test: pd.DataFrame,
        model: GBRegressor | GBClassifier,
        target_transformer: RobustScaler | LabelEncoder
    ) -> None:
        """Serializes the model and exports all associated run artifacts.

        This method creates a timestamped directory and saves the trained
        model weights, a metadata JSON containing performance metrics and
        configurations, and a CSV of predictions generated from the provided
        test data.

        Args:
            gradient_boosting_config (dict): Hyperparameters for the boosting
                algorithm.
            weak_learner_config (dict): Hyperparameters for the weak learners
            model (GBRegressor | GBClassifier): The trained gradient boosting
                model instance.
            predictions (np.ndarray): Test set predictions.
            probabilities (np.ndarray): Test set probability predictions, if
                problem type is classification.
        """
        model_path = self.model_dir / f"{model.end_timestamp}"
        model_path.mkdir(parents=True, exist_ok=True)

        model.save_model(str(model_path / "model.joblib"))

        if self.problem_type == "classification":
            best_score = {
                "best_log_loss": round(model.best_log_loss, 6),
                "best_accuracy": round(model.best_accuracy, 6)
            }
        
        else:
            best_score = {
                "best_mse": round(model.best_mse, 6),
                "best_r2": round(model.best_r2, 6)
            }

        info = {
            "run_metadata": {
                "problem_type": self.problem_type,
                "best_iter": model.best_iter,
                **best_score,
                "start_timestamp": model.start_timestamp,
                "end_timestamp": model.end_timestamp
            },
            "gradient_boosting_config": gradient_boosting_config,
            "weak_learner_config": weak_learner_config,
            }
            
        with open(str(model_path / "info.json"), 'w', encoding='utf-8') as file:
            json.dump(info, file, indent=4)

        predictions = model.predict(test)
        
        predictions_output = pd.Series(
            target_transformer.inverse_transform(predictions),
            index=test.index,
            name=self.target
        )
        
        predictions_output.to_csv(str(model_path / "predictions.csv"))

        if self.problem_type == "classification":
            probabilities = model.predict_proba(test)
            
            column_names = self.target_transformer.classes_
            probabilities_output = pd.DataFrame(
                probabilities,
                index=test.index,
                columns=column_names
            )

            probabilities_output.to_csv(str(model_path / "probabilities.csv"))
