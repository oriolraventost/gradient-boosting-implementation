import numpy as np

from numba import njit, prange
from typing import Union, Any

TreeType = Union[dict[str, Any], float]

class XGBTree:
    """A single decision tree for Gradient Boosting using histogram-based splits.
    
    This implementation utilizes second-order Taylor expansion information 
    (gradients and hessians) to optimize a custom loss function. It employs 
    a histogram-based algorithm to accelerate the discovery of the best 
    split point, significantly reducing the computational complexity from 
    sorting-based methods.

    Attributes:
        max_depth (int): Maximum depth of the tree.
        reg_lambda (float): L2 regularization term on leaf weights (prevents overfitting).
        gamma (float): Minimum gain required to make a further partition on a leaf node.
        structure (TreeType | None): Nested dictionary representation of the tree 
            after fitting.
    """

    def __init__(self, max_depth: int, reg_lambda: float, gamma: float):
        """Initializes hyperparameters for the tree growth."""
        self.max_depth: int = max_depth
        self.reg_lambda: float = reg_lambda
        self.gamma: float = gamma
        self.structure: TreeType | None = None

    def _build_histograms(self, g: np.ndarray, h: np.ndarray, X_binned: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Wrapper to call the high-performance static histogram builder.
        
        Args:
            g (np.ndarray): First-order gradients.
            h (np.ndarray): Second-order gradients (hessians).
            X_binned (np.ndarray): Discretized (binned) feature matrix.

        Returns:
            tuple[np.ndarray, np.ndarray]: Aggregated gradient and hessian histograms.
        """
        return self._build_histograms_static(g, h, X_binned, 256)

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _build_histograms_static(g: np.ndarray, h: np.ndarray, X_binned: np.ndarray, n_bins: int):
        """Numba-accelerated histogram construction.
        
        This method parallelizes across features to populate gradient and 
        hessian bins, which are later used for calculating split gain.
        """
        n_samples, n_feat = X_binned.shape
        hist_g = np.zeros((n_feat, n_bins), dtype=np.float64)
        hist_h = np.zeros((n_feat, n_bins), dtype=np.float64)
        
        for j in prange(n_feat):
            for i in range(n_samples):
                bin_idx = int(X_binned[i, j])
                hist_g[j, bin_idx] += g[i]
                hist_h[j, bin_idx] += h[i]
                
        return hist_g, hist_h

    def _find_best_split(
        self, 
        g: np.ndarray, 
        h: np.ndarray, 
        X_binned: np.ndarray
    ) -> tuple[int, int] | None:
        """Finds the (feature_index, bin_index) split that maximizes the Gain.
        
        Returns:
            tuple[int, int] | None: The feature and bin index for the split, 
                or None if no gain exceeds the gamma threshold.
        """
        hist_g, hist_h = self._build_histograms(g, h, X_binned)
        
        G_tot: float = np.sum(g)
        H_tot: float = np.sum(h)
        
        G_L = np.cumsum(hist_g, axis=1)[:, :-1]
        H_L = np.cumsum(hist_h, axis=1)[:, :-1]
        G_R = G_tot - G_L
        H_R = H_tot - H_L

        gain = 0.5 * (
            (G_L**2 / (H_L + self.reg_lambda)) + 
            (G_R**2 / (H_R + self.reg_lambda)) - 
            (G_tot**2 / (H_tot + self.reg_lambda))
        ) - self.gamma
        
        if np.max(gain) <= 0:
            return None
        
        f_idx_local, bin_idx = np.unravel_index(np.argmax(gain), gain.shape)
        return int(f_idx_local), int(bin_idx)

    def _grow_tree(
        self, 
        g: np.ndarray, 
        h: np.ndarray, 
        X_binned: np.ndarray, 
        depth: int, 
        active_cols: np.ndarray
    ) -> TreeType:
        """Recursive method to build the tree structure using a greedy approach.
                
        Args:
            g (np.ndarray): Gradients of samples in the current node.
            h (np.ndarray): Hessians of samples in the current node.
            X_binned (np.ndarray): Binned feature matrix for the samples.
            depth (int): Current recursion depth.
            active_cols (np.ndarray): Mapping of local feature indices to 
                original feature indices.

        Returns:
            TreeType: A dictionary for internal nodes or a float for leaf nodes.
        """
        if depth >= self.max_depth or len(g) < 2:
            return float(-np.sum(g) / (np.sum(h) + self.reg_lambda))

        split = self._find_best_split(g, h, X_binned)
        
        if split is None:
            return float(-np.sum(g) / (np.sum(h) + self.reg_lambda))

        f_idx_local, b_idx = split
        mask = X_binned[:, f_idx_local] <= b_idx
        
        return {
            'feat': active_cols[f_idx_local],
            'bin': b_idx,
            'left': self._grow_tree(g[mask], h[mask], X_binned[mask], depth + 1, active_cols),
            'right': self._grow_tree(g[~mask], h[~mask], X_binned[~mask], depth + 1, active_cols)
        }

    def fit(self, X_binned_subsampled: np.ndarray, g: np.ndarray, h: np.ndarray, active_cols: np.ndarray) -> None:
        """Fits the tree on a column-subsampled binned dataset.
        
        Args:
            X_binned_subsampled (np.ndarray): Feature matrix containing only 
                subsampled columns.
            g (np.ndarray): Global gradients.
            h (np.ndarray): Global hessians.
            active_cols (np.ndarray): Indices of columns included in the subsample.
        """
        self.structure = self._grow_tree(g, h, X_binned_subsampled, 0, active_cols)

    def predict(self, X_binned_full: np.ndarray) -> np.ndarray:
        """Predicts for all samples using the full discretized matrix.
        
        Args:
            X_binned_full (np.ndarray): The full binned dataset.

        Returns:
            np.ndarray: Predicted values (leaf weights) for each sample.
        """
        return self._predict_recursive(X_binned_full, self.structure)

    def _predict_recursive(self, X_binned: np.ndarray, node: TreeType) -> np.ndarray:
        """Internal recursive traversal to determine leaf node values."""
        if not isinstance(node, dict):
            return np.full(X_binned.shape[0], node)
        
        res = np.zeros(X_binned.shape[0])
        mask = X_binned[:, node['feat']] <= node['bin']
        
        if np.any(mask):
            res[mask] = self._predict_recursive(X_binned[mask], node['left'])
        if np.any(~mask):
            res[~mask] = self._predict_recursive(X_binned[~mask], node['right'])
        return res