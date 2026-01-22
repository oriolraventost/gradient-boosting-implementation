import pandas as pd
import argparse
import os

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.config.main import metadata

class Processor:
    """Handles end-to-end data transformation."""
    def __init__(self):
        self.preprocessor: ColumnTransformer | None = None

    def _setup_columns(self, data: pd.DataFrame, target: str, id_column: str, ordinal_map: dict):
        """Identifies column types dynamically from the dataframe."""
        categorical_cols = [col for col in data.select_dtypes(include=['object']).columns 
                           if col not in ordinal_map and col != target]
        numerical_cols = [col for col in data.select_dtypes(include=['number']).columns 
                         if col not in [target, id_column]]
        return categorical_cols, numerical_cols

    def _create_transformer(self, cat_cols: list, num_cols: list, ord_map: dict):
        """Constructs the Scikit-Learn transformer object."""
        return ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='median'), num_cols),
                ('ord', Pipeline([
                    ('impute', SimpleImputer(strategy='constant', fill_value='NA')),
                    ('encode', OrdinalEncoder(categories=list(ord_map.values())))
                ]), list(ord_map.keys())),
                ('nom', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
            ]
        ).set_output(transform="pandas")

    def run(self, dataset: str):
        """Orchestrates the loading, transformation, and saving of data."""
        df = pd.read_csv(f"data/raw/{dataset}")
        
        dataset_name = os.path.splitext(dataset)[0].removesuffix("_train").removesuffix("_test")

        target = metadata[dataset_name]["target"]
        id_column = metadata[dataset_name]["id_column"]
        ordinal_map = metadata[dataset_name]["ordinal_map"]

        cat_cols, num_cols = self._setup_columns(df, target, id_column, ordinal_map)
        self.preprocessor = self._create_transformer(cat_cols, num_cols, ordinal_map)
        
        processed_df = self.preprocessor.fit_transform(df)
        processed_df[target] = df[target].values
        
        processed_df.to_csv(f"data/processed/{dataset}", index=False)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    
    processor = Processor()
    processor.run(dataset=args.dataset)