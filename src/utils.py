import numpy as np

def derivative_mean_squared_error(y_train: np.ndarray, train_preds: np.ndarray) -> np.ndarray:
        """Compute the gradient of the mean squared error loss with respect to predictions."""
        return 2 * (train_preds - y_train)

def shift_plus_one(x):
    return x + 1