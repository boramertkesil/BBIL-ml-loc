import numpy as np
from src.models.base_model import BaseModel
from src.metrics.distances import pairwise_distances

class kNNLocalizer(BaseModel):
    """
    k-Nearest Neighbors model for indoor localization.

    Uses RSSI fingerprints and predicts XY coordinates by averaging
    the closest training samples.
    """
    def __init__(self, k=5, metric="euclidean", layers=None):
        """
        Parameters
        ----------
        k : int
            Number of nearest neighbors.
        metric : str or callable, default "euclidean"
            Distance metric used for neighbor search.
        layers : list[BaseLayer], optional
            Preprocessing layers applied before fitting and prediction.
        """
        super().__init__(layers=layers)
        self.k = k

        self.metric = metric
        self.X_train = None
        self.Y_train = None

    def _fit(self, X, y):
        """
        Stores training data.
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
        dists = pairwise_distances(X, self.X_train, metric=self.metric)

        M = X.shape[0]
        preds = np.zeros((M, 2), dtype=float)

        for j in range(M):

            nn_idx = np.argpartition(dists[j], self.k)[:self.k]
            nn_y = self.Y_train[nn_idx]

            preds[j] = np.mean(nn_y, axis=0)

        return preds