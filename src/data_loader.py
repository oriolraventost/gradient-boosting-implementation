import pandas as pd
import json

from pathlib import Path
from sklearn.model_selection import train_test_split

from src import *
from src.gb_regressor import GBRegressor
from src.gb_classifier import GBClassifier

class DataLoader:
    """Handles the ingestion and persistence of dataset files.

    This class manages the file system interactions, specifically loading
    raw data from various formats and saving/loading  preprocessed versions.

    Attributes:
        raw_dir (Path): Directory where original dataset files are stored.
        processed_dir (Path): Directory where processed CSVs are saved.
        model_dir (Path): Directory where trained models are saved.
        id_column (str): The column name used as the identifier.
        target (str): The column name for the target variable.
        problem_type (str): Regression or classification.
    """

    def __init__(
        self,
        dataset_name: str,
        id_column: str,
        target_column: str,
        problem_type: str
    ):
        """Initializes the DataLoader with project-specific paths."""
        self.raw_dir = Path(DATA_PATH) / f"{dataset_name}/raw"
        
        self.processed_dir = Path(DATA_PATH) / f"{dataset_name}/processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_dir = Path(MODELS_PATH) / f"{dataset_name}"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.id_column = id_column
        self.target_column = target_column
        self.problem_type = problem_type
    
    def load_raw_data(
        self
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads raw data by detecting either combined or split
        feature/label files.

        Returns:
            A tuple containing training, validation and test datasets.
        """
        train_path = self.raw_dir / "train.csv"
        test_path = self.raw_dir / "test.csv"

        train_features_path = self.raw_dir / "train_features.csv"
        train_labels_path = self.raw_dir / "train_labels.csv"
        test_features_path = self.raw_dir / "test_features.csv"

        if train_path.exists() and test_path.exists():
            full_train = pd.read_csv(train_path, index_col=self.id_column)
            test = pd.read_csv(test_path, index_col=self.id_column)
        
        elif (train_features_path.exists() and
            train_labels_path.exists() and
            test_features_path.exists()):

            train_features = pd.read_csv(
                train_features_path,
                index_col=self.id_column
            )

            train_labels = pd.read_csv(
                train_labels_path,
                index_col=self.id_column
            )
            
            test = pd.read_csv(
                test_features_path,
                index_col=self.id_column
            )

            full_train = pd.merge(
                train_features,
                train_labels,
                left_index=True,
                right_index=True
            )

        else:
            raise FileNotFoundError(
                "Missing raw data files in the expected directory."
            )

        stratify_col = (
            full_train[self.target_column]
            if self.problem_type == "classification"
            else None
        )

        train, valid = train_test_split(
            full_train,
            test_size=0.2,
            stratify=stratify_col,
            random_state=42
        )
        
        return train, valid, test
    
    def save_processed_data(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        test: pd.DataFrame
    ) -> None:
        """Persists processed DataFrames to the disk.

        Args:
            train (pd.DataFrame): The cleaned training DataFrame.
            valid (pd.DataFrame): The cleaned validation DataFrame.
            test (pd.DataFrame): The cleaned test DataFrame.
        """
        train.to_csv(self.processed_dir / "train.csv")
        valid.to_csv(self.processed_dir / "valid.csv")
        test.to_csv(self.processed_dir / "test.csv")

    def load_processed_data(
        self
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads previously saved processed data.

        Returns:
            A tuple containing processed training, validation and
                test datasets.
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
    
    def save_model(
        self,
        model: GBRegressor | GBClassifier,
        test_data: pd.DataFrame,
        start_timestamp: str,
        end_timestamp: str,
        gradient_boosting_config: dict,
        weak_learner_config: dict
    ) -> None:
        """Serializes the model and exports all associated run artifacts.

        This method creates a timestamped directory and saves the trained
        model weights, a metadata JSON containing performance metrics and
        configurations, and a CSV of predictions generated from the provided
        test data.

        Args:
            model (GBRegressor | GBClassifier): The trained gradient boosting
                model instance.
            test_data (pd.DataFrame): The test dataset used to generate final
                predictions.
            start_timestamp (str): String indicating when training began.
            end_timestamp (str): String indicating when training finished.
            gradient_boosting_config (dict): Hyperparameters for the boosting
                algorithm.
            weak_learner_config (dict): Hyperparameters for the weak learners.
        """
        model_path = self.model_dir / f"{end_timestamp}"
        model_path.mkdir(parents=True, exist_ok=True)


        model.save_model(str(model_path / "model.joblib"))

        if isinstance(model, GBClassifier):
            problem_type = "classification"
            best_score = {
                "best_ce": round(model.best_ce, 6),
                "best_accuracy": round(model.best_accuracy, 6)
            }
        
        else:
            problem_type = "regression"
            best_score = {"best_mse": round(model.best_mse, 6)}

        info = {
            "run_metadata": {
                "problem_type": problem_type,
                "best_iter": model.best_iter,
                **best_score,
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp
            },
            "gradient_boosting_config": gradient_boosting_config,
            "weak_learner_config": weak_learner_config,
            }
            
        with open(str(model_path / "info.json"), 'w', encoding='utf-8') as file:
            json.dump(info, file, indent=4)

        preds = model.predict(test_data)
    
        output = pd.DataFrame({
            self.id_column: test_data.index,
            self.target_column: preds
        })

        output.to_csv(str(model_path / "test_preds.csv"), index=False)
