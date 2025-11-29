import numpy as np
from src.models.base_model import BaseModel

class kNNLocalizer(BaseModel):
    """
    k-Nearest Neighbors model for indoor localization.

    Uses RSSI fingerprints and predicts XY coordinates by averaging
    the closest training samples.
    """

    def __init__(self, k=5, layers=None):
        """
        Parameters
        ----------
        k : int
            Number of nearest neighbors.
        layers : list[BaseLayer], optional
            Preprocessing layers applied before fitting and prediction.
        """
        super().__init__(layers=layers)
        self.k = k

        self.X_train = None
        self.Y_train = None


    def _dist(self, A, B):
        """
        Compute full pairwise Euclidean distance matrix between two 2D NumPy arrays.

        Parameters
        ----------
        A : np.ndarray, shape (N, D)
        B : np.ndarray, shape (M, D)

        Returns
        -------
        dists : np.ndarray, shape (N, M)
        """
        A_sq = np.sum(A**2, axis=1).reshape(-1, 1)
        B_sq = np.sum(B**2, axis=1).reshape(1, -1)
        cross = A @ B.T

        return np.sqrt(A_sq + B_sq - 2 * cross)


    def _fit(self, X, y):
        """
        Stores training data. Preprocessing is handled in BaseModel.fit().
        """
        self.X_train = X
        self.Y_train = y
        return self

    def _predict(self, X):
        """
        Parameters
        ----------
        X : np.ndarray of shape (M, D)

        Returns
        -------
        preds : np.ndarray (M, 2)
        """
        dists = self._dist(self.X_train, X)

        M = X.shape[0]
        _X = np.zeros((M, 2), dtype=float)

        for j in range(M):

            nn_idx = np.argpartition(dists[:, j], self.k)[:self.k]
            nn_y = self.Y_train[nn_idx]  # shape (k, 2)

            _X[j] = np.mean(nn_y, axis=0)

        return _X