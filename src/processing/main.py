import argparse
import logging
import pandas as pd

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.config.main import metadata

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Processor:
    """Handles end-to-end data transformation for Train/Test sets."""
    def __init__(self):
        self.preprocessor: ColumnTransformer | None = None

    @staticmethod
    def _merge_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
        """Concatenates training and test data with a source indicator."""
        return pd.concat([
            train_data.assign(is_train=True),
            test_data.assign(is_train=False)
        ], axis=0, ignore_index=True)

    @staticmethod
    def _setup_columns(data: pd.DataFrame, target: str, id_column: str, ordinal_map: dict):
        """Dynamically categorizes columns excluding special utility columns."""
        exclude = {target, id_column, 'is_train'}
        
        cat_cols = [c for c in data.select_dtypes(include=['object']).columns 
                   if c not in ordinal_map and c not in exclude]
        
        num_cols = [c for c in data.select_dtypes(include=['number']).columns 
                   if c not in exclude]
                   
        return cat_cols, num_cols

    def _create_transformer(self, cat_cols: list, num_cols: list, ord_map: dict) -> ColumnTransformer:
        """Constructs the Scikit-Learn transformer."""
        return ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='median'), num_cols),
                ('ord', Pipeline([
                    ('impute', SimpleImputer(strategy='constant', fill_value='NA')),
                    ('encode', OrdinalEncoder(categories=list(ord_map.values())))
                ]), list(ord_map.keys())),
                ('nom', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
            ],
            verbose_feature_names_out=False
        ).set_output(transform="pandas")

    def run(self, train_file: str, test_file: str) -> None:
        """
        Orchestrates loading, fitting (on train only), transforming, and saving.
        """
        raw_path = Path("data/raw")
        out_path = Path("data/processed")
        out_path.mkdir(parents=True, exist_ok=True)

        train_df = pd.read_csv(raw_path / train_file)
        test_df = pd.read_csv(raw_path / test_file)
        
        dataset_name = Path(train_file).stem.replace("_train", "")
        meta = metadata.get(dataset_name)
        
        if not meta:
            raise ValueError(f"No metadata found for dataset: {dataset_name}")

        target = meta["target"]
        id_col = meta["id_column"]
        ord_map = meta.get("ordinal_map", {})

        df = self._merge_data(train_df, test_df)
        cat_cols, num_cols = self._setup_columns(df, target, id_col, ord_map)
        
        self.preprocessor = self._create_transformer(cat_cols, num_cols, ord_map)
        
        logger.info(f"Fitting preprocessor on {len(train_df)} training samples...")
        self.preprocessor.fit(df[df['is_train']])
        
        processed_df = self.preprocessor.transform(df)
        processed_df[target] = df[target]

        train_processed = processed_df[df['is_train']]
        test_processed = processed_df[~df['is_train']].drop(columns=[target])

        train_out = out_path / train_file
        test_out = out_path / test_file

        train_processed.to_csv(train_out, index=False)
        test_processed.to_csv(test_out, index=False)

        logger.info(f"Saved processed train ({train_processed.shape}) to {train_out}")
        logger.info(f"Saved processed test ({test_processed.shape}) to {test_out}")
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    args = parser.parse_args()
    
    processor = Processor()
    processor.run(train_file=args.train_file, test_file=args.test_file)