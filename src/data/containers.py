from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class DataContainer(ABC):
    """
    Base class for holding a part of the dataset.

    Stores a dataframe and exposes X and y as NumPy arrays.
    Subclasses define which columns belong to features and targets.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataframe for this container.

    Attributes
    ----------
    df : pd.DataFrame
        Raw dataframe.
    suffix : str
        File suffix used when loading related CSV files.
    X : np.ndarray
        Feature values.
    y : np.ndarray or None
        Target values, or None if not defined.
    """
    df: pd.DataFrame
    suffix = None

    def __init__(self, df: pd.DataFrame):
        self.df = df

    @classmethod
    def get_csv_suffix(cls):
        return f"_{cls.suffix}.csv"

    @property
    @abstractmethod
    def X_columns(self) -> list[str]:
        pass

    @property
    @abstractmethod
    def y_columns(self) -> list[str]:
        pass

    def to_numpy(self, cols: list[str]):
        return self.df[cols].to_numpy(np.float32)

    @property
    def X(self):
        return self.to_numpy(self.X_columns)

    @property
    def y(self):
        if not self.y_columns:
            return None
        return self.to_numpy(self.y_columns)

class RSSI(DataContainer):
    suffix = 'data_wide'

    @property
    def X_columns(self):
        return [col for col in self.df.columns if col.startswith("edge_")]

    @property
    def y_columns(self):
        return ["realx", "realy"]
    
class Acc(DataContainer):
    suffix = 'acc'

    @property
    def X_columns(self):
        return ["accx","accy","accz"]

    @property
    def y_columns(self):
        return None

class Pos(DataContainer):
    suffix = 'pos'

    @property
    def X_columns(self):
        return ["realx", "realy"]

    @property
    def y_columns(self):
        return None

class Com(DataContainer):
    suffix = 'com'

    @property
    def X_columns(self):
        return ["azimuth"]

    @property
    def y_columns(self):
        return None


