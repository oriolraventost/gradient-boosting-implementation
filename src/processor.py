import logging
import numpy as np
import pandas as pd

from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    MinMaxScaler
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Processor:
    """Orchestrates end-to-end data transformation for Train, Validation,
    and Test sets.

    This class automates feature discovery and transformation using
    Scikit-Learn pipelines. It manages the transition from raw tabular or
    image data to high-performance NumPy arrays, ensuring that scaling and
    encoding parameters are derived strictly from the training set.

    Attributes:
        target (str): Name of the target variable column.
        id_column (str): Name of the unique identifier column.
        problem_type (str): The learning task.
        cat_cols (list[str] | None): Names of categorical features.
        num_cols (list[str] | None): Names of numerical features.
        feature_transformer (ColumnTransformer): Pipeline for transforming
            features.
        target_transformer (MinMaxScaler | OrdinalEncoder): Pipeline for
            transforming targets.
    """

    def __init__(
        self,
        id_column: str,
        target: str,
        problem_type: str,
        cat_cols: list[str] | None = None,
        num_cols: list[str] | None = None,
    ):
        """Initializes the processor."""
        self.target: str = target
        self.id_column: str = id_column
        self.problem_type: str = problem_type
        
        self.cat_cols: list[str] | None = cat_cols
        self.num_cols: list[str] | None = num_cols
        
        self.feature_transformer: ColumnTransformer = (
            self._get_feature_transformer()
        )

        self.target_transformer: MinMaxScaler | OrdinalEncoder = (
            self._get_target_transformer()
        )
    
    def fit_transform(
        self,
        data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fits transformers and transforms the input dataset.

        Learns preprocessing parameters (medians, scales, categories) from 
        the provided dataframe and applies the transformations to both 
        features and the target variable.

        Args:
            data: The raw input DataFrame containing both features and target.

        Returns:
            pd.DataFrame: A processed DataFrame with scaled numeric values, 
                encoded categorical features, and a transformed target.
        """
        processed_data = self.feature_transformer.fit_transform(
            data
        )
        processed_data.index = data.index
        processed_data[self.target] = self.target_transformer.fit_transform(
            data[self.target].values.reshape(-1, 1)
        ).ravel()

        return processed_data

    def transform(
        self,
        data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Applies pre-fitted transformations to the input dataset.

        Uses the parameters (medians, scales, categories) previously learned 
        during the fit process to transform features and the target variable. 
        This ensures consistency between training and inference.

        Args:
            data: The raw input DataFrame containing features and the target.

        Returns:
            pd.DataFrame: The processed DataFrame with transformations applied 
                to all columns and the original index preserved.
        """
        processed_data = self.feature_transformer.transform(
            data
        )
        processed_data.index = data.index
        processed_data[self.target] = self.target_transformer.transform(
            data[self.target].values.reshape(-1, 1)
        ).ravel()

        return processed_data

    def split_features_target(
        self,
        data: pd.DataFrame | Subset
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extracts X and y, handling both Tabular and Computer Vision
        formats.

        For images, this performs 0-1 normalization and converts to
        (C, H, W) format. For tabular data, it separates the configured
        target column.

        Args:
            data (pd.DataFrame | Subset): Input data container.

        Returns:
            tuple[np.ndarray, np.ndarray]: Features (X) and Labels (y).
        """
        if self.problem_type == "computer_vision":
            X = data.dataset.data[data.indices]
            y = np.array(data.dataset.targets)[data.indices]
            
            if not isinstance(X, np.ndarray):
                X = X.numpy()

            X = X.astype(np.float32) / 255.0

            if X.ndim == 3:
                X = np.expand_dims(X, -1)
            
            X = X.transpose(0, 3, 1, 2)
            
            y = np.array(y).astype(np.float32)
        
        else:
            X = data.drop(columns=[self.target]).values
            y = data[self.target].values
        
        return X, y

    def convert_to_numpy(self, data: pd.DataFrame | Subset) -> np.ndarray:
        """Converts various data structures into a unified NumPy array format.

        This method handles extraction from PyTorch Subsets (common in 
        computer vision) and pandas DataFrames (common in tabular tasks).

        Args:
            data (pd.DataFrame | Subset): The input data structure. 
                Can be a pandas DataFrame or a torch.utils.data.Subset.

        Returns:
            np.ndarray: The data converted to a NumPy array.
        """
        if isinstance(data, Subset):
            data = data.dataset.data[data.indices]
        
            if not isinstance(data, np.ndarray):
                data = data.numpy()
            
            data = data.astype(np.float32) / 255.0
            
            if data.ndim == 3:
                data = np.expand_dims(data, -1)
            
            data = data.transpose(0, 3, 1, 2)
        
        else:
            data = data.to_numpy()
        
        return data
    
    def _get_feature_transformer(self) -> ColumnTransformer:
        """Initializes the preprocessing pipeline for features.

        Numeric features are scaled using a MinMaxScaler. Categorical
        features are one-hot encoded. Outputs a pandas DataFrame.

        Returns:
            ColumnTransformer: A scikit-learn transformer configured for 
                the dataset's numeric and categorical columns.
        """
        cat_pipeline = Pipeline([
            ('encode', OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False
            ))
        ])

        num_pipeline = Pipeline([
            ('scale', MinMaxScaler())
        ])

        return ColumnTransformer(
            transformers=[
                ("cat", cat_pipeline, self.cat_cols),
                ("num", num_pipeline, self.num_cols),
            ],
            verbose_feature_names_out=False
        ).set_output(transform="pandas")
    
    def _get_target_transformer(self) -> MinMaxScaler | OrdinalEncoder:
        """Selects the appropriate scaler or encoder for the target variable.

        Uses a MinMax for regression tasks to handle outliers or an 
        OrdinalEncoder for classification tasks.

        Returns:
            MinMaxScaler | OrdinalEncoder: The transformer corresponding 
                to the specified problem type.
        """
        if self.problem_type == "regression":
            return MinMaxScaler()
        
        else:
            return OrdinalEncoder()
    
    def cut_data(
        self,
        data: pd.DataFrame,
        max_rows: int = 100_000
    ) -> pd.DataFrame:
        """Reduces the dataset size. If the dataset exceeds the limit,
        it performs a stratified reduction for classification problems
        to maintain class proportions. For regression, it performs a
        simple random shuffle and cut.

        Args:
            data (pd.DataFrame): The input dataframe to be processed.
            max_rows (int): Maximum number of rows of reduced data.

        Returns:
            pd.DataFrame: A subset of the input data.
        """
        if len(data) <= max_rows:
            return data
        
        stratify_col = (
            data[self.target]
            if self.problem_type == "classification"
            else None
        )

        _, data_subset = train_test_split(
            data,
            test_size=max_rows,
            stratify=stratify_col,
            random_state=42
        )
        
        return data_subset