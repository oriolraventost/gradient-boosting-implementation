import numpy as np

from numba import njit, prange
from typing import Union, Any

# Type Alias for the recursive structure
TreeType = Union[dict[str, Any], float]

class XGBTree:
    """
    A single decision tree for Gradient Boosting.
    """
    def __init__(self, max_depth: int, reg_lambda: float, gamma: float):
        self.max_depth: int = max_depth
        self.reg_lambda: float = reg_lambda
        self.gamma: float = gamma
        self.structure: TreeType | None = None

    def _build_histograms(self, g: np.ndarray, h: np.ndarray, X_binned: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # We call the static method and pass the max_bin (e.g., 256)
        return self._build_histograms_static(g, h, X_binned, 256)

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _build_histograms_static(g: np.ndarray, h: np.ndarray, X_binned: np.ndarray, n_bins: int):
        n_samples, n_feat = X_binned.shape
        hist_g = np.zeros((n_feat, n_bins), dtype=np.float64)
        hist_h = np.zeros((n_feat, n_bins), dtype=np.float64)
        
        # Parallelize over features (the outer loop)
        for j in prange(n_feat):
            for i in range(n_samples):
                # Ensure bin_idx is an integer for indexing
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
        """
        Finds the best (feature_index, bin_index) split for the provided data.
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
        """
        Recursive method to build the tree structure.
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
        """
        Fits the tree on a column-subsampled binned dataset.
        """
        self.structure = self._grow_tree(g, h, X_binned_subsampled, 0, active_cols)

    def predict(self, X_binned_full: np.ndarray) -> np.ndarray:
        """
        Predicts for all samples using the full (global) X_binned matrix.
        """
        return self._predict_recursive(X_binned_full, self.structure)

    def _predict_recursive(self, X_binned: np.ndarray, node: TreeType) -> np.ndarray:
        if not isinstance(node, dict):
            return np.full(X_binned.shape[0], node)
        
        res = np.zeros(X_binned.shape[0])
        mask = X_binned[:, node['feat']] <= node['bin']
        
        if np.any(mask):
            res[mask] = self._predict_recursive(X_binned[mask], node['left'])
        if np.any(~mask):
            res[~mask] = self._predict_recursive(X_binned[~mask], node['right'])
        return res