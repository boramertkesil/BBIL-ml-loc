from abc import ABC, abstractmethod

class BaseLayer(ABC):
    """
    Base class for preprocessing layers.

    Layers process input data before it is passed to a model.
    Subclasses must implement the transform method.
    """

    def fit(self, X, y=None):
        """
        Optional: layers that learn parameters should override this.
        """
        return self

    @abstractmethod
    def transform(self, X):
        """
        Required: transforms the input data and returns the result.
        """
        pass

    """ 
    May use later
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    """