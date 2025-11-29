import numpy as np
from src.preprocessing.base_layer import BaseLayer

class KalmanFilterLayer(BaseLayer):
    """
    Apply a 1D Kalman filter to each column of a 2D array.

    Each column gets its own filter instance.
    The filter smooths RSSI values by reducing noise.

    Parameters
    ----------
    Q : float, default 0.04
        Process noise.
    R : float, default 4.0
        Measurement noise.
    """
    def __init__(self, Q=0.04, R=4.0):
        self.Q = Q
        self.R = R
        
        self.filters = None

    # Helper method for creating filter for each column
    def _create_filters(self, num_cols):
        return {col: KalmanFilter1D(Q=self.Q, R=self.R) for col in range(num_cols)}

    def transform(self, X):
        """
        Smooth all columns of a 2D array using Kalman filtering.

        Parameters
        ----------
        X : np.ndarray of shape (N, M)
            Input matrix with M columns.

        Returns
        -------
        np.ndarray
            Smoothed matrix with the same shape.
        """
        if X.ndim != 2:
            raise ValueError("KalmanFilter1D expects a 2D NumPy array of shape (N, M).")

        n_samples, n_cols = X.shape

        # Create filters on first call
        if self.filters is None:
            self.filters = self._create_filters(n_cols)

        _X = np.zeros_like(X)

        # Apply each filter to its column
        for col in range(n_cols):
            filt = self.filters[col]
            for i in range(n_samples):
                _X[i, col] = filt.update(X[i, col])

        return _X

# Very nice explanation of the use case of Kalman Filter over RSSI values:
# https://www.wouterbulten.nl/posts/kalman-filters-explained-removing-noise-from-rssi-signals/

class KalmanFilter1D:
    def __init__(self, Q=0.5, R=4.0):
        self.Q = Q
        self.R = R
        self.x = None
        self.P = None

    def update(self, z):
        # --- Initialization ---
        if self.x is None:
            if np.isnan(z):
                return np.nan
            self.x = z
            self.P = 10.0
            return self.x

        # --- Predict ---
        self.P = self.P + self.Q

        # Skip updated if NaN
        if np.isnan(z):
            return self.x

        # --- Update ---
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * self.P

        return self.x