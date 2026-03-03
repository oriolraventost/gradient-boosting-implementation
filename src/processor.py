import logging
import pandas as pd

from category_encoders import CountEncoder, CatBoostEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, TargetEncoder

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
        target_column (str): Name of target column.
        id_column (str): Name of ID column.
        high_card_cat_cols (list[str] | None): List of high cardinality 
            categorical feature names.
        low_card_cat_cols (list[str] | None): List of low cardinality 
            categorical feature names.
        num_cols (list[str] | None): List of numerical feature names.
        transformer (ColumnTransformer | None): The Scikit-Learn pipeline.
    """

    def __init__(self, target_column: str, id_column: str):
        """Initializes the processor."""
        self.target_column = target_column
        self.id_column = id_column
        self.cat_cols: list[str] | None = None
        self.num_cols: list[str] | None = None
        self.transformer: ColumnTransformer | None = None
    
    def run(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Orchestrates fitting (on train only) and transforming data.
        
        This method follows the "no data leakage" principle by fitting the 
        transformer strictly on the training data and only transforming the
        validation and test data.

        Args:
            train (pd.DataFrame): The raw training dataset.
            valid (pd.DataFrame): The raw validation dataset.
            test (pd.DataFrame): The raw testing dataset.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple
                containing the processed training, validation and
                testing DataFrames.
        """
        logger.info("Processing pipeline starting...")
        
        self._get_cat_cols(train)
        self._get_num_cols(train)

        train[self.high_card_cat_cols] = train[self.high_card_cat_cols].astype(str)
        valid[self.high_card_cat_cols] = valid[self.high_card_cat_cols].astype(str)
        test[self.high_card_cat_cols] = test[self.high_card_cat_cols].astype(str)

        train[self.low_card_cat_cols] = train[self.low_card_cat_cols].astype(str)
        valid[self.low_card_cat_cols] = valid[self.low_card_cat_cols].astype(str)
        test[self.low_card_cat_cols] = test[self.low_card_cat_cols].astype(str)

        self._get_transformer()
        
        logger.info("Fitting and transforming training data...")
        processed_train = pd.DataFrame(
            self.transformer.fit_transform(
                train.drop(columns=[self.target_column]),
                train[self.target_column]
            ), 
            index=train.index,
            columns=self.transformer.get_feature_names_out()
        )

        logger.info("Transforming validation data...")
        processed_valid = pd.DataFrame(
            self.transformer.transform(valid), 
            index=valid.index,
            columns=self.transformer.get_feature_names_out()
        )

        logger.info("Transforming test data...")
        processed_test = pd.DataFrame(
            self.transformer.transform(test), 
            index=test.index,
            columns=self.transformer.get_feature_names_out()
        )

        processed_train = processed_train.copy()
        processed_valid = processed_valid.copy()
        processed_test = processed_test.copy()

        self._drop_duplicate_and_constant(
            processed_train,
            processed_valid,
            processed_test
        )

        processed_train[self.target_column] = train[self.target_column]
        processed_valid[self.target_column] = valid[self.target_column]

        return processed_train, processed_valid, processed_test

    def _get_cat_cols(self, data: pd.DataFrame) -> None:
        """Identifies categorical columns based on object and boolean types.
        
        Args:
            data (pd.DataFrame): The input dataframe to scan.
        """
        cat_cols = [
            col for col in data.select_dtypes(include=["object", "bool"])
            if col not in [self.id_column, self.target_column]
        ]

        n_unique_categories = data[cat_cols].nunique()

        self.high_card_cat_cols = n_unique_categories[n_unique_categories > 255].index
        self.low_card_cat_cols = n_unique_categories[n_unique_categories <= 255].index

        if cat_cols:
            logger.info(
                f"Categorical columns detected: {', '.join(cat_cols)}"
            )

    def _get_num_cols(self, data: pd.DataFrame) -> None:
        """Identifies numerical columns by excluding objects and booleans.
        
        Args:
            data (pd.DataFrame): The input dataframe to scan.
        """
        self.num_cols = [
            col for col in data.select_dtypes(exclude=["object", "bool"])
            if col not in [self.id_column, self.target_column]
        ]
        
        if self.num_cols:
            logger.info(
                f"Numerical columns detected: {', '.join(self.num_cols)}"
            )


    def _get_transformer(self) -> None:
        """Initialize the feature engineering pipeline for numeric and
        categorical attributes.

        The transformation logic follows a two-pronged strategy:
        - Numerical: Imputes missing values using MICE and applies quantile
            transformation to normal distribution.
        - Categorical: Imputes missing values with additional category,
            target and frequency encodes and transform numerical results
            to normal distribution.

        Sets the transformer to output pandas DataFrames to maintain 
        feature name transparency throughout the pipeline.
        """
        high_card_cat_pipeline = Pipeline([
            ('impute', SimpleImputer(
                strategy='constant',
                fill_value="missing"
            )),
            ('encode', TargetEncoder(cv=20, target_type='continuous')),
            ('scale', RobustScaler())
        ])

        low_card_cat_pipeline = Pipeline([
            ('encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ])

        num_pipeline = Pipeline([
            ('impute', IterativeImputer()),
            ('scale', RobustScaler())
        ])

        self.transformer = ColumnTransformer(
            transformers=[
                ("high_card_cat", high_card_cat_pipeline, self.high_card_cat_cols),
                ("low_card_cat", low_card_cat_pipeline, self.low_card_cat_cols),
                ("num", num_pipeline, self.num_cols),
            ],
            verbose_feature_names_out=False
        ).set_output(transform="default")

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
            train (pd.DataFrame): Training feature set.
            valid (pd.DataFrame): Validation feature set.
            test (pd.DataFrame): Test feature set.
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