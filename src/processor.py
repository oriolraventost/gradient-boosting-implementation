import logging
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Processor:
    """Handles end-to-end data transformation for Train/Test sets."""
    def __init__(self):
        """Initializes the processor with empty column lists and transformer."""
        self.cat_cols: list[str] | None = None
        self.num_cols: list[str] | None = None

        self.transformer: ColumnTransformer | None = None
        
    def _get_cat_cols(self, data: pd.DataFrame, target: str, id: str):
        """Identifies categorical columns based on object and boolean types."""
        self.cat_cols = [col for col in data.select_dtypes(include=["object", "bool"]) if col not in [target, id]]
        
    def _get_num_cols(self, data: pd.DataFrame, target: str, id: str):
        """Identifies numerical columns by excluding objects and booleans."""
        self.num_cols = [col for col in data.select_dtypes(exclude=["object", "bool"]) if col not in [target, id]]

    def _get_column_transformer(self):
        """Constructs a pipeline that median-imputes numerical features and one-hot encodes categorical features."""
        self.transformer = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('impute', SimpleImputer(strategy='median'))
                ]), self.num_cols),
                ('cat', Pipeline([
                    ('encode', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
                ]), self.cat_cols),
            ],
            verbose_feature_names_out=False
        ).set_output(transform="pandas")

    def run(self, train_data: pd.DataFrame, test_data: pd.DataFrame, target: str, id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Orchestrates fitting (on train only) and transforming both sets."""
        logger.info("Processing pipeline starting...")
        
        self._get_cat_cols(data=train_data, target=target, id=id)
        if self.cat_cols:
            logger.info(f"Categorical columns detected: {" ".join(self.cat_cols)}")

        self._get_num_cols(data=train_data, target=target, id=id)
        if self.num_cols:
            logger.info(f"Numerical columns detected: {" ".join(self.num_cols)}")

        self._get_column_transformer()
        
        logger.info("Fitting and transforming training data...")
        processed_train_data = self.transformer.fit_transform(train_data)
        processed_train_data[target] = train_data[target]

        logger.info("Transforming test data...")
        processed_test_data = self.transformer.transform(test_data)

        return processed_train_data, processed_test_data
