from dataclasses import dataclass
from typing import Optional
from typing import Tuple

import numpy as np


def rbf_kernel(
    X: np.ndarray, Y: np.ndarray, lengthscale: float, variance: float
) -> np.ndarray:
    """Squared-exponential (RBF) kernel.

    K(x, y) = variance * exp(-0.5 * ||x - y||^2 / lengthscale^2)
    """
    X2 = np.sum(X**2, axis=1).reshape(-1, 1)
    Y2 = np.sum(Y**2, axis=1).reshape(1, -1)
    d2 = X2 - 2 * X @ Y.T + Y2
    return variance * np.exp(-0.5 * d2 / (lengthscale**2 + 1e-12))


@dataclass
class GP:
    """Minimal Gaussian Process regressor with an RBF kernel."""

    lengthscale: float = 1.0
    variance: float = 1.0
    noise: float = 1e-6

    _X: Optional[np.ndarray] = None
    _y: Optional[np.ndarray] = None
    _L: Optional[np.ndarray] = None
    _alpha: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit GP hyperparameters to data using Cholesky factorization."""
        self._X = np.array(X, dtype=float)
        self._y = np.array(y, dtype=float).reshape(-1, 1)
        K = rbf_kernel(self._X, self._X, self.lengthscale, self.variance)
        K[np.diag_indices_from(K)] += self.noise
        L = np.linalg.cholesky(K)
        v = np.linalg.solve(L, self._y)
        alpha = np.linalg.solve(L.T, v)
        self._L = L
        self._alpha = alpha

    def predict(self, Xstar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict posterior mean and variance at test points Xstar."""
        assert self._X is not None, "Call fit() before predict()"
        assert self._L is not None, "Call fit() before predict()"
        assert self._alpha is not None, "Call fit() before predict()"
        Kxs = rbf_kernel(self._X, Xstar, self.lengthscale, self.variance)
        mean = Kxs.T @ self._alpha
        w = np.linalg.solve(self._L, Kxs)
        kss = np.full((Xstar.shape[0],), self.variance)
        var = kss - np.sum(w**2, axis=0)
        var = np.maximum(var, 1e-12)
        return mean.ravel(), var
