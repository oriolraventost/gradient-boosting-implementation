import logging
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    MinMaxScaler
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
        cat_cols(list[str] | None): List of categorical feature names.
        num_cols(list[str] | None): List of numerical feature names.
        transformer(ColumnTransformer | None): The Scikit-Learn pipeline.
    """

    def __init__(self, target: str, id_column: str):
        """Initializes the processor."""
        self.target = target
        self.id_column = id_column
        self.cat_cols: list[str] | None = None
        self.num_cols: list[str] | None = None
        self.transformer: ColumnTransformer | None = None
    
    def run(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Orchestrates fitting (on train only) and transforming both sets.
        
        This method follows the "no data leakage" principle by fitting the 
        transformer strictly on the training data and only transforming the
        test data.

        Args:
            train_data (pd.DataFrame): The raw training dataset.
            test_data (pd.DataFrame): The raw testing dataset.
            target (str): The column name of the dependent variable.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the
                processed training and testing DataFrames.
        """
        logger.info("Processing pipeline starting...")
        
        self._get_cat_cols(train_data)
        self._get_num_cols(train_data)

        train_data[self.cat_cols] = train_data[self.cat_cols].astype(str)
        test_data[self.cat_cols] = test_data[self.cat_cols].astype(str)

        self._get_transformer()
        
        logger.info("Fitting and transforming training data...")
        processed_train_data = self.transformer.fit_transform(
            train_data
        )
        processed_train_data.index = train_data.index
        processed_train_data[self.target] = train_data[self.target]

        logger.info("Transforming test data...")
        processed_test_data = self.transformer.transform(test_data)
        processed_test_data.index = test_data.index

        constant_cols = [
            col for col in processed_train_data.columns 
            if processed_train_data[col].nunique() <= 1
        ]
        
        if constant_cols:
            logger.info(
                f"Removing constant columns: {", ".join(constant_cols)}"
            )
            
            processed_train_data.drop(
                columns=constant_cols,
                inplace=True
            )
            processed_test_data.drop(
                columns=constant_cols,
                inplace=True
            )

        return processed_train_data, processed_test_data

    def _get_cat_cols(self, data: pd.DataFrame) -> None:
        """Identifies categorical columns based on object and boolean types.
        
        Args:
            data (pd.DataFrame): The input dataframe to scan.
        """
        self.cat_cols = [
            col for col in data.select_dtypes(include=["object", "bool"])
            if col not in [self.id_column, self.target]
        ]
        
        if self.cat_cols:
            logger.info(
                f"Categorical columns detected: {', '.join(self.cat_cols)}"
            )

    def _get_num_cols(self, data: pd.DataFrame) -> None:
        """Identifies numerical columns by excluding objects and booleans.
        
        Args:
            data (pd.DataFrame): The input dataframe to scan.
        """
        self.num_cols = [
            col for col in data.select_dtypes(exclude=["object", "bool"])
            if col not in [self.id_column, self.target]
        ]
        
        if self.num_cols:
            logger.info(
                f"Numerical columns detected: {', '.join(self.num_cols)}"
            )


    def _get_transformer(self) -> None:
        """Initialize the feature engineering pipeline for numeric and
        categorical attributes.

        The transformation logic follows a two-pronged strategy:
        - Numerical: Imputes missing values using the median and applies 
          robust scaling.
        - Categorical: Performs One-Hot Encoding restricted to categories
          with frequency higher than 1%.

        Sets the transformer to output pandas DataFrames to maintain 
        feature name transparency throughout the pipeline.
        """
        cat_pipeline = Pipeline([
            ('encode', OneHotEncoder(
                min_frequency=0.01,
                handle_unknown='ignore',
                sparse_output=False
            ))
        ])

        num_pipeline = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', MinMaxScaler())
        ])

        self.transformer = ColumnTransformer(
            transformers=[
                ("cat", cat_pipeline, self.cat_cols),
                ("num", num_pipeline, self.num_cols),
            ],
            verbose_feature_names_out=False
        ).set_output(transform="pandas")
