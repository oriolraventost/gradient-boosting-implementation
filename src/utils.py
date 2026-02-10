import numpy as np
import pandas as pd

from sklearn.utils.extmath import softmax

def derivative_mean_squared_error(y_train: np.ndarray, train_preds: np.ndarray) -> np.ndarray:
    """Compute the gradient of the Mean Squared Error (MSE) loss.

    Args:
        y_train (np.ndarray): Ground truth target values.
        train_preds (np.ndarray): Current ensemble predictions (logits/scores).

    Returns:
        np.ndarray: The element-wise gradient used as pseudo-residuals for 
            the next regression tree.
    """
    return 2 * (train_preds - y_train)


def derivative_log_loss(y_train: pd.Series, train_preds: pd.DataFrame) -> pd.DataFrame:
    """Compute the gradient of the Cross-Entropy (Log-Loss) for multi-class classification.

    Args:
        y_train (pd.Series): Categorical ground truth labels.
        train_preds (pd.DataFrame): Current raw scores (logits) from the ensemble, 
            where columns represent distinct classes.

    Returns:
        pd.DataFrame: A matrix of gradients with shape (n_samples, n_classes), 
            aligned with the input index and columns.
    """
    y_train_one_hot = pd.get_dummies(y_train).reindex(
        columns=range(train_preds.shape[1]),
        fill_value=0
    ).values

    probability_preds = softmax(train_preds.values)
    
    gradients = probability_preds - y_train_one_hot
    
    return pd.DataFrame(
        gradients, 
        index=train_preds.index, 
        columns=train_preds.columns
    )