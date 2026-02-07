import numpy as np
import pandas as pd

from sklearn.utils.extmath import softmax

def derivative_mean_squared_error(y_train: np.ndarray, train_preds: np.ndarray) -> np.ndarray:
        """Compute the gradient of the mean squared error loss with respect to predictions."""
        return 2 * (train_preds - y_train)

def derivative_log_loss(y_train: pd.Series, train_preds: pd.DataFrame) -> pd.DataFrame:
        """Compute the gradient of the cross entropy loss with respect to predictions, for each class."""        
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