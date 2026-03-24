import numpy as np
import pandas as pd
import json
import yaml

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from torchvision import datasets
from torch.utils.data import Subset

from src import *
from src.gb_regressor import GBRegressor
from src.gb_classifier import GBClassifier

class DataManager:
    """Handles the ingestion, preprocessing, and persistence of data
    artifacts.

    This class manages the directory structure for raw and processed
    datasets, loads configuration settings, and handles the serialization of
    trained models and their associated predictions.

    Attributes:
        raw_dir (Path | None): Directory containing the original CSV datasets.
        processed_dir (Path | None): Directory for storing cleaned/transformed
            data.
        model_dir (Path | None): Base directory for saving model run
            artifacts.
        id_column (str | None): Column name representing the unique
            identifier.
        target (str | None): Name of the target variable column.
        problem_type (str | None): Category of ML task.
        test_index (range | None): Index range used for identifying test
            predictions.
    """

    def __init__(self):
        """Initializes the DataManager attributes."""
        self.raw_dir: Path | None = None
        self.processed_dir: Path | None = None
        self.model_dir: Path | None = None

        self.id_column: str | None = None
        self.target: str | None = None
        self.problem_type: str | None = None

        self.test_index: list[int] | None = None

    def load_config(self) -> tuple[dict, dict]:
        """Loads YAML configuration files and sets up the project directory
        tree.

        Returns:
            tuple[dict, dict]: Dataset configurations and modeling
                hyperparameters.
        """
        with open(DATASETS_CONFIG_PATH, "r") as f:
            datasets_config = yaml.safe_load(f)
        
        with open(MODELING_CONFIG_PATH, "r") as f:
            modeling_config = yaml.safe_load(f)
        
        active_dataset = modeling_config["main"]["active_dataset"]

        self.id_column = datasets_config[active_dataset]["id_column"]
        self.target = datasets_config[active_dataset]["target"]
        self.problem_type = datasets_config[active_dataset]["problem_type"]
        
        if self.problem_type != "computer_vision":
            self.raw_dir = Path(DATA_PATH) / active_dataset / "raw"
            
            self.processed_dir = Path(DATA_PATH) / active_dataset / "processed"
            self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_dir = Path(MODELS_PATH) / active_dataset
        self.model_dir.mkdir(parents=True, exist_ok=True)

        return datasets_config, modeling_config
    
    def load_raw_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Loads the initial training and test CSV files.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Training and test DataFrames.
        """
        train_path = self.raw_dir / "train.csv"
        test_path = self.raw_dir / "test.csv"

        train = pd.read_csv(train_path, index_col=self.id_column)
        test = pd.read_csv(test_path, index_col=self.id_column)

        train_size = len(train)
        test_size = len(test)

        self.test_index = range(train_size, train_size + test_size)
        
        return train, test
    
    def save_processed_data(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        test: pd.DataFrame
    ) -> None:
        """Saves processed datasets to the 'processed' directory.

        Args:
            train (pd.DataFrame): Transformed training data.
            valid (pd.DataFrame): Transformed validation data.
            test (pd.DataFrame): Transformed test data.
        """
        train.to_csv(self.processed_dir / "train.csv")
        valid.to_csv(self.processed_dir / "valid.csv")
        test.to_csv(self.processed_dir / "test.csv")

    def load_processed_data(
        self
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Retrieves previously saved processed datasets.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Processed
                split data.
        """
        train = pd.read_csv(
            self.processed_dir / "train.csv",
            index_col=self.id_column
        )

        valid = pd.read_csv(
            self.processed_dir / "valid.csv",
            index_col=self.id_column
        )

        test = pd.read_csv(
            self.processed_dir / "test.csv",
            index_col=self.id_column
        )

        return train, valid, test
    
    def load_image_data(
        self,
        active_dataset: str
    ) -> tuple[Subset, Subset, np.ndarray]:
        """Loads torchvision datasets and creates a random
        train/validation split.

        Args:
            active_dataset (str): Name of the dataset.

        Returns:
            tuple: Training Subset, Validation Subset, and Test Subset.
        """
        image_data_map = {
            "mnist": datasets.MNIST,
            "cifar10": datasets.CIFAR10
        }

        image_data_class = image_data_map[active_dataset]

        full_train_set = image_data_class(
            root=f"{DATA_PATH}/{active_dataset}",
            train=True,
            download=True
        )

        targets = full_train_set.targets

        train_idx, valid_idx = train_test_split(
            range(len(targets)),
            test_size=0.2,
            stratify=targets,
            random_state=42
        )

        train_set = Subset(full_train_set, train_idx)
        valid_set = Subset(full_train_set, valid_idx)
        
        test_set = image_data_class(
            root=f"{DATA_PATH}/{active_dataset}",
            train=False,
            download=True
        )

        test_set = Subset(
            test_set, 
            list(range(len(test_set)))
        )

        full_train_size = len(full_train_set)
        test_size = len(test_set)

        self.test_index = range(full_train_size, full_train_size + test_size)

        return train_set, valid_set, test_set

    def save_model(
        self,
        gradient_boosting_config: dict,
        weak_learner_config: dict,
        test: pd.DataFrame,
        model: GBRegressor | GBClassifier,
        target_transformer: RobustScaler | OrdinalEncoder
    ) -> None:
        """Exports model weights, metadata, and predictions to a unique
        run directory.

        Args:
            gradient_boosting_config (dict): Global boosting parameters.
            weak_learner_config (dict): Individual learner parameters.
            test (pd.DataFrame): Test features for final inference.
            model (Any): The trained GBRegressor or GBClassifier.
            target_transformer (RobustScaler | OrdinalEncoder): Transformer
                used for inverse-scaling targets.
        """
        model_path = self.model_dir / f"{model.end_timestamp}"
        model_path.mkdir(parents=True, exist_ok=True)

        model.save_model(str(model_path / "model.joblib"))

        if self.problem_type == "regression":
            best_score = {
                "best_mse": round(model.best_mse, 6),
                "best_r2": round(model.best_r2, 6)
            }
        
        else:
            best_score = {
                "best_log_loss": round(model.best_log_loss, 6),
                "best_accuracy": round(model.best_accuracy, 6)
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
            
        with open(
            str(model_path / "info.json"), 'w', encoding='utf-8'
        ) as file:
            json.dump(info, file, indent=4)

        predictions = model.predict(test)

        predictions_output = pd.Series(
            target_transformer.inverse_transform(
                predictions.reshape(-1, 1)
            ).ravel(),
            index=self.test_index,
            name=self.target
        )

        predictions_output.index.name = self.id_column
        predictions_output.to_csv(str(model_path / "predictions.csv"))

        if self.problem_type != "regression":
            probabilities = model.predict_proba(test)
            
            column_names = target_transformer.categories_[-1]
            probabilities_output = pd.DataFrame(
                probabilities,
                index=self.test_index,
                columns=column_names
            )

            probabilities_output.index.name = self.id_column
            probabilities_output.to_csv(str(model_path / "probabilities.csv"))
