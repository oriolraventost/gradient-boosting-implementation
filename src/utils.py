import numpy as np
import pandas as pd

from sklearn.utils.extmath import softmax
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler

def derivative_mean_squared_error(
    y_train: np.ndarray,
    train_preds: np.ndarray
) -> np.ndarray:
    """Compute the gradient of the Mean Squared Error (MSE) loss.

    Args:
        y_train (np.ndarray): Ground truth target values.
        train_preds (np.ndarray): Current ensemble predictions.

    Returns:
        np.ndarray: The element-wise gradient.
    """
    return 2 * (train_preds - y_train)


def derivative_log_loss(
    y_train: np.ndarray,
    train_preds: np.ndarray
) -> np.ndarray:
    """Compute the gradient of the CE loss for classification.

    Args:
        y_train (np.ndarray): Categorical ground truth labels.
        train_preds (np.ndarray): Current logits from the ensemble,
            where columns represent distinct classes.

    Returns:
        np.ndarray: The element-wise gradient.
    """
    y_train_one_hot = np.zeros(train_preds.shape)
    y_train_one_hot[np.arange(len(y_train)), y_train.astype(int)] = 1

    probability_preds = softmax(train_preds)
    
    return probability_preds - y_train_one_hot

def preprocess(
    train_features: pd.DataFrame,
    train_labels: pd.DataFrame,
    test_features: pd.DataFrame,
    target: str,
    id: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Handles end-to-end data transformation for Train/Test sets.
    
    This class automates feature discovery by splitting columns into
    categorical and numerical types, then applies a Scikit-Learn
    ColumnTransformer pipeline to handle imputation and encoding.
        
    This method follows the "no data leakage" principle by fitting the 
    transformer strictly on the training data and only transforming the
    test data. Numerical features' missing values are imputed with the
    median and categorical features are ordinal encoded.

    Args:
        train_features (pd.DataFrame): The raw training features.
        train_labels (pd.DataFrame): The raw training labels.
        test_features (pd.DataFrame): The raw test features.
        target (str): The column name of the dependent variable.
        id_cols (str): ID column name.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the
            processed training and testing DataFrames.
    """
    train_data = pd.merge(train_features, train_labels, on=id)
    test_data = test_features
    
    cat_cols = [
        col for col
        in train_data.select_dtypes(include=["object", "bool"])
        if col not in [target, id]

    ]

    num_cols = [
        col for col
        in train_data.select_dtypes(exclude=["object", "bool"])
        if col not in [target, id]
    ]

    train_data[cat_cols] = train_data[cat_cols].astype(str)
    test_data[cat_cols] = test_data[cat_cols].astype(str)

    transformer = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('impute', SimpleImputer(strategy='median')),
                ('scale', StandardScaler())
            ]), num_cols),
            ('cat', Pipeline([
                ('impute', SimpleImputer(
                    strategy="constant", fill_value='NA'
                )),
                ('encode', OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=-1
                ))
            ]), cat_cols),
        ],
        verbose_feature_names_out=False
    ).set_output(transform="pandas")
    
    processed_train_data = transformer.fit_transform(train_data)
    processed_train_data.index = train_data.index

    processed_test_data = transformer.transform(test_data)
    processed_test_data.index = test_data.index

    le = LabelEncoder()
    processed_train_data[target] = le.fit_transform(train_data[target])

    return processed_train_data, processed_test_data