import logging
import numpy as np
import pandas as pd

from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    RobustScaler
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Processor:
    """Handles end-to-end data transformation for Train/Test sets.
    
    This class automates feature discovery by splitting columns into
    categorical and numerical types, then applies a Scikit-Learn
    ColumnTransformer pipeline to handle imputation and encoding.

    Attributes:
        target (str): Name of target column.
        id_column (str): Name of ID column.
        problem_type (str): Learning task (regression or classification).
        cat_cols (list[str] | None): List of categorical feature names.
        num_cols (list[str] | None): List of numerical feature names.
        feature_transformer (ColumnTransformer | None): The Scikit-Learn
            pipeline for transforming feature variables.
        target_transformer (RobustScaler | LabelEncoder | None): The
            Scikit-Learn object for transforming the target variable.
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
        
        self.feature_transformer: ColumnTransformer | None = None
        self.target_transformer: RobustScaler | OrdinalEncoder | None = None
    
    def transform_features(
        self,
        full_train_data: pd.DataFrame,
        test_data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Orchestrates fitting and transforming both sets.
        
        This method follows the "no data leakage" principle by fitting the 
        transformer strictly on the training data and only transforming the
        validation and test data.

        Args:
            full_train_data (pd.DataFrame): The raw full training dataset.
            test_data (pd.DataFrame): The raw testing dataset.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple
                containing the processed training, validation and
                testing DataFrames.
        """
        logger.info("Processing pipeline starting...")

        self._get_feature_transformer()
        
        full_train_data = self._cut_data(full_train_data)

        train_data, valid_data = self._train_valid_split(
            full_train_data
        )
        
        logger.info("Fitting and transforming training data...")
        processed_train_data = self.feature_transformer.fit_transform(
            train_data
        )
        processed_train_data.index = train_data.index
        processed_train_data[self.target] = train_data[self.target]

        logger.info("Transforming validation data...")
        processed_valid_data = self.feature_transformer.transform(valid_data)
        processed_valid_data.index = valid_data.index
        processed_valid_data[self.target] = valid_data[self.target]

        logger.info("Transforming test data...")
        processed_test_data = self.feature_transformer.transform(test_data)
        processed_test_data.index = test_data.index

        self._drop_duplicate_and_constant(
            processed_train_data,
            processed_valid_data,
            processed_test_data
        )

        return processed_train_data, processed_valid_data, processed_test_data

    def transform_target(
        self,
        y_train: np.ndarray,
        y_valid: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """ Fits the target transformer and transforms training and
        validation targets.

        Selects RobustScaler for regression or OrdinalEncoder for
        classification. Handles the necessary 2D reshaping for the
        transformers and flattens the output back to 1D arrays.

        Args:
            y_train (np.ndarray): Training target values.
            y_valid (np.ndarray): Validation target values.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the
                transformed arrays.
        """
        if self.problem_type == "regression":
            self.target_transformer = RobustScaler()
        
        else:
            self.target_transformer = OrdinalEncoder()
        
        processed_y_train = self.target_transformer.fit_transform(
            y_train.reshape(-1, 1)
        ).ravel()
        
        processed_y_valid = self.target_transformer.transform(
            y_valid.reshape(-1, 1)
        ).ravel()

        return processed_y_train, processed_y_valid

    def split_features_target(
        self,
        data: pd.DataFrame | Subset
    ) -> tuple[np.ndarray, np.ndarray]:
        """Splits input data into feature and target arrays.

        Handles both computer vision datasets and tabular pandas DataFrames.
        For image data, it performs normalization and adds a channel dimension
        if necessary. For tabular data, it extracts the target column based on
        the configured target attribute.

        Args:
            data (pd.DataFrame | Subset): The input data. If problem_type is 
                computer_vision, this is expected to be a Torchvision 
                Subset object. Otherwise, a pandas DataFrame.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the feature and
                target arrays.
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
            
            if data.ndim == 3:
                data = np.expand_dims(data, -1)
            
            data = data.transpose(0, 3, 1, 2)
        
        else:
            data = data.to_numpy()
        
        return data
    
    def _get_feature_transformer(self):
        """Initialize the feature engineering pipeline for numeric and
        categorical attributes.

        The transformation logic follows a two-pronged strategy. On
        numerical features, imputes missing values using the median
        and applies robust scaling. On categorical features, performs
        One-Hot Encoding restricted to categories with frequency higher
        than 1%.
        """
        cat_pipeline = Pipeline([
            ('encode', OneHotEncoder(
                drop="if_binary",
                min_frequency=0.01,
                handle_unknown='ignore',
                sparse_output=False
            ))
        ])

        num_pipeline = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', RobustScaler())
        ])

        self.feature_transformer = ColumnTransformer(
            transformers=[
                ("cat", cat_pipeline, self.cat_cols),
                ("num", num_pipeline, self.num_cols),
            ],
            verbose_feature_names_out=False
        ).set_output(transform="pandas")
    
    def _cut_data(
        self,
        data: pd.DataFrame,
        max_rows: int = 100_000,
        iqr_factor: float = 1.5
    ) -> pd.DataFrame:
        """Reduces the dataset size. If the dataset exceeds the limit,
        it performs a stratified reduction for classification problems
        to maintain class proportions. For regression, it performs a
        simple random shuffle and cut, after removing mild target outliers.

        Args:
            data (pd.DataFrame): The input dataframe to be processed.
            max_rows (int): Maximum number of rows of reduced data.
            iqr_factor (float): Outlier removal IQR factor.

        Returns:
            pd.DataFrame: A subset of the input data.
        """
        if self.problem_type == "regression":
            q1 = data[self.target].quantile(0.25)
            q3 = data[self.target].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_factor * iqr
            upper = q3 + iqr_factor * iqr
            data = data[(data[self.target] >= lower) & (data[self.target] <= upper)]

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

    def _train_valid_split(
        self,
        data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Splits the dataset into training and validation sets.

        Performs a stratified split if the problem type is classification to
        ensure class proportions are maintained across folds. Otherwise, 
        performs a standard split.

        Args:
            data (pd.DataFrame): The complete processed dataset containing 
                both features and the target column.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the 
                split at a 80/20 ratio.
        """
        stratify_col = (
            data[self.target] if self.problem_type == "classification"
            else None
        )

        return train_test_split(
            data, 
            stratify=stratify_col,
            test_size=0.2,
            random_state=42
        )

    def _drop_duplicate_and_constant(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        test: pd.DataFrame
    ) -> None:
        """Removes non-informative columns and training duplicates
        to prevent bias.

        Identifies constant columns based on the training set and removes
        them from all splits. Drops duplicate rows from the training set
        only to ensure the model doesn't overfit to repeated observations.

        Args:
            train (pd.DataFrame): Training data.
            valid (pd.DataFrame): Validation data.
            test (pd.DataFrame): Test data.
        """
        constant_cols = [
            col for col in train.columns 
            if train[col].nunique(dropna=False) <= 1
        ]

        if constant_cols:
            logger.info(f"Removing {len(constant_cols)} constant columns.")
            train.drop(columns=constant_cols, inplace=True)
            valid.drop(columns=constant_cols, inplace=True)
            test.drop(columns=constant_cols, inplace=True)

        initial_rows = len(train)
        train.drop_duplicates(inplace=True)
        dropped_rows = initial_rows - len(train)
        
        if dropped_rows > 0:
            logger.info(
                f"Dropped {dropped_rows} duplicate rows from training set."
            )