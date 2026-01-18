import numpy as np

def derivative_mean_squared_error(y: np.array, y_preds: np.array) -> np.array:
    """Compute the gradient of the mean squared error loss with respect to predictions."""
    return 2 * (y_preds - y)