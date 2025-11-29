from abc import ABC, abstractmethod
from src.preprocessing.base_layer import BaseLayer

class BaseModel(ABC):
    """
    Base class for machine learning models.

    Handles preprocessing and defines a abstract method for fit and predict.
    Subclasses implement the internal _fit and _predict methods.

    Parameters
    ----------
    layers : list[BaseLayer], optional
        Preprocessing layers applied before training and prediction.
    """
    def __init__(self, layers: list[BaseLayer] = None):
        self.layers = layers or []

    def _apply_preprocessing(self, X):
        for layer in self.layers:
            X = layer.transform(X)
        return X

    def fit(self, X, y=None):
        X = self._apply_preprocessing(X)
        return self._fit(X, y)

    def predict(self, X):
        X = self._apply_preprocessing(X)
        return self._predict(X)

    @abstractmethod
    def _fit(self, X, y):
        pass

    @abstractmethod
    def _predict(self, X):
        pass