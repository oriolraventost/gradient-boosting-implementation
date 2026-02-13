import pandas as pd

from pathlib import Path

class DataLoader:
    """Handles the ingestion and persistence of dataset files.

    This class manages the file system interactions, specifically loading
    raw data from various formats and saving/loading  preprocessed versions.

    Attributes:
        raw_dir (Path): Directory where original dataset files are stored.
        processed_dir (Path): Directory where processed CSVs are saved.
        id (str): The column name used as the identifier.
    """

    def __init__(self, dataset_name: str, id_column: str):
        """Initializes the DataLoader with project-specific paths.

        Args:
            dataset_name: The folder name for the specific dataset.
            id: The name of the ID column to be used as the DataFrame index.
        """
        self.raw_dir = Path(f"data/{dataset_name}/raw")
        self.processed_dir = Path(f"data/{dataset_name}/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.id_column = id_column
    
    def load_raw_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Loads raw data by detecting either combined or split
        feature/label files.

        It first checks for 'train.csv'. If not found, it attempts to merge 
        'train_features.csv' and 'train_labels.csv' on the ID index.

        Returns:
            A tuple containing (train_df, test_df).

        Raises:
            FileNotFoundError: If neither the combined nor the split files
                exist.
        """
        train_path = self.raw_dir / "train.csv"
        test_path = self.raw_dir / "test.csv"

        if train_path.exists() and test_path.exists():
            train = pd.read_csv(train_path, index_col=self.id_column)
            test = pd.read_csv(test_path, index_col=self.id_column)
            return train, test
        
        train_features_path = self.raw_dir / "train_features.csv"
        train_labels_path = self.raw_dir / "train_labels.csv"
        test_features_path = self.raw_dir / "test_features.csv"

        if not (train_features_path.exists() and train_labels_path.exists()):
            raise FileNotFoundError(
                "Missing raw data files in the expected directory."
            )

        train_features = pd.read_csv(
            train_features_path,
            index_col=self.id_column
        )

        train_labels = pd.read_csv(
            train_labels_path,
            index_col=self.id_column
        )
        
        test_features = pd.read_csv(
            test_features_path,
            index_col=self.id_column
        )

        train = pd.merge(
            train_features,
            train_labels,
            left_index=True,
            right_index=True
        )
        
        return train, test_features
    
    def save_processed_data(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame
    ) -> None:
        """Persists processed DataFrames to the disk.

        Args:
            train: The cleaned training DataFrame.
            test: The cleaned test DataFrame.
        """
        train.to_csv(self.processed_dir / "train.csv")
        test.to_csv(self.processed_dir / "test.csv")

    def load_processed_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Loads previously saved processed data.

        Returns:
            A tuple containing the processed (train_df, test_df).
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