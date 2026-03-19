import numpy as np
import pandas as pd
import json
import yaml

from pathlib import Path
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from torchvision import datasets
from torch.utils.data import Subset, random_split

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
        test_index (list[int] | None): List of indeces for test predictions.
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
        """Loads the configuration from a YAML file.

        Returns:
            tuple[dict, dict]: Two dictionaries containing datasets and
                modeling configuration settings.
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
        """Loads raw dataset CSV files as pandas DataFrames.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Raw training and test data.
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
        """Persists processed DataFrames to the disk.

        Args:
            train (pd.DataFrame): The cleaned training data.
            valid (pd.DataFrame): The cleaned validation data.
            test (pd.DataFrame): The cleaned test data.
        """
        train.to_csv(self.processed_dir / "train.csv")
        valid.to_csv(self.processed_dir / "valid.csv")
        test.to_csv(self.processed_dir / "test.csv")

    def load_processed_data(
        self
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads previously saved processed data.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Processed
                training, validation and test data.
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
        """Loads image data and performs a random train-validation split.

        This method identifies the requested dataset, downloads it to a 
        local directory if not already present, and splits the official 
        training set into training and validation subsets using an 80/20
        ratio.

        Args:
            active_dataset (str): The name of the dataset to load. 
                Supported values are 'mnist' and 'cifar10'.

        Returns:
            tuple: A tuple containing the training, validation and test
                data, the latter without the target.
        """
        image_data_map = {
            "MNIST": datasets.MNIST,
            "cifar10": datasets.CIFAR10
        }

        image_data_class = image_data_map[active_dataset]

        full_train_set = image_data_class(
            root=f"{DATA_PATH}/{active_dataset}",
            train=True,
            download=True
        )

        train_size = int(0.8 * len(full_train_set))
        valid_size = len(full_train_set) - train_size
        
        train_set, valid_set = random_split(
           full_train_set, 
            [train_size, valid_size]
        )
        
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
        """Serializes the model and exports all associated run artifacts.

        This method creates a timestamped directory and saves the trained
        model weights, a metadata JSON containing performance metrics and
        configurations, and a CSV of predictions generated from the provided
        test data.

        Args:
            gradient_boosting_config (dict): Hyperparameters for the gradient
                boosting algorithm.
            weak_learner_config (dict): Hyperparameters for the weak learners.
            test (pd.DataFrame): Test data to generate predictions on.
            model (GBRegressor | GBClassifier): The trained gradient boosting
                model instance.
            target_transformer (RobustScaler | OrdinalEncoder): The target
                transformer to apply the inverse transformation.
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
