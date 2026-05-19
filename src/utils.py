import numpy as np

from sklearn.utils.extmath import softmax

def first_derivative_mean_squared_error(
    y_train: np.ndarray,
    train_preds: np.ndarray
) -> np.ndarray:
    """Compute the gradient of the Mean Squared Error (MSE) loss.

    Args:
        y_train (np.ndarray): Ground truth target values.
        train_preds (np.ndarray): Current ensemble predictions.

    Returns:
        np.ndarray: The element-wise gradient vector.
    """
    return 2 * (train_preds - y_train)


def first_derivative_log_loss(
    y_train: np.ndarray,
    train_preds: np.ndarray
) -> np.ndarray:
    """Compute the gradient of the CE loss for classification.

    Args:
        y_train (np.ndarray): Categorical ground truth labels.
        train_preds (np.ndarray): Current logits from the ensemble,
            where columns represent distinct classes.

    Returns:
        np.ndarray: The element-wise gradient matrix.
    """
    y_train_one_hot = np.zeros(train_preds.shape)
    y_train_one_hot[np.arange(len(y_train)), y_train.astype(int)] = 1

    probability_preds = softmax(train_preds)
    
    return probability_preds - y_train_one_hot


def second_derivative_mean_squared_error(
    y_train: np.ndarray,
    train_preds: np.ndarray
) -> np.ndarray:
    """Compute the diagonal of the Hessian for Mean Squared Error.
    
    Args:
        y_train (np.ndarray): Ground truth target values.
        train_preds (np.ndarray): Current ensemble predictions.

    Returns:
        np.ndarray: The diagonal array elements of the Hessian matrix.
    """
    return np.full_like(train_preds, 2.0)


def second_derivative_log_loss(
    y_train: np.ndarray,
    train_preds: np.ndarray
) -> np.ndarray:
    """Compute the diagonal of the Hessian for Softmax Cross-Entropy.
    
    Args:
        y_train (np.ndarray): Categorical ground truth labels.
        train_preds (np.ndarray): Current logits from the ensemble,
            where columns represent distinct classes.

    Returns:
        np.ndarray: The diagonal array elements of the Hessian matrix.
    """
    probability_preds = softmax(train_preds)
    
    return probability_preds * (1 - probability_preds)