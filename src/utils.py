import numpy as np

from sklearn.utils.extmath import softmax

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