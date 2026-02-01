import logging
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, FunctionTransformer

from src.utils import shift_plus_one

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Processor:
    """Handles end-to-end data transformation for Train/Test sets."""
    def __init__(self, dataset_config: dict):
        self.target: str = dataset_config["target"]
        
        self.transformer: ColumnTransformer = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('impute', SimpleImputer(strategy='median')),
                    ('scale', StandardScaler())
                ]), dataset_config["num_cols"]),
                ('ord', Pipeline([
                    ('impute', SimpleImputer(strategy='constant', fill_value='NA')),
                    ('encode', OrdinalEncoder())
                ]), dataset_config["cat_cols"]),
            ],
            verbose_feature_names_out=False
        ).set_output(transform="pandas")

    def run(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Orchestrates fitting (on train only) and transforming.
        """
        logger.info(f"Fitting preprocessor on {len(train_data)} training samples...")
        
        processed_train_data = self.transformer.fit_transform(train_data)
        processed_train_data[self.target] = train_data[self.target]
        
        processed_test_data = self.transformer.transform(test_data)

        return processed_train_data, processed_test_data
