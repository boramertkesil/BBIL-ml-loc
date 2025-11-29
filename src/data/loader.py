from typing import TypeVar, Type, Dict
from src.data.containers import DataContainer
from src.data.partitions import Partition
import pandas as pd
import os
import re

def get_all_prefixes(dirname: str):
    """
    Find all filename prefixes for ``*_data.csv`` files in a directory.

    Parameters
    ----------
    dirname : str
        Directory to scan.

    Returns
    -------
    list of str
        List of filename prefixes.
    """
    match_prefix = re.compile(r'(?P<prefix>.+)_data\.csv')
    chunks = [f for f in os.listdir(dirname) if f.endswith("_data.csv")]
    prefixes = []

    for chunk in chunks:
        m = match_prefix.match(chunk)
        if m:
            prefixes.append(m.group('prefix'))
        else:
            print(f"Skipping unrecognized filename: {chunk}")

    return prefixes

T = TypeVar("T", bound=DataContainer)

def load_partition(dirname: str, partition: Partition, dtypes: list[Type[T]]) -> Dict[Type[T], T]:
    """
    Load a single dataset partition.

    Reads all CSV files under the given split (train/test/valid)
    and builds containers for each desired data type.

    Parameters
    ----------
    dirname : str
        Base dataset directory.
    partition : Partition
        Which split to load.
    dtypes : list of DataContainer types
        Containers to construct.

    Returns
    -------
    dict
        Mapping of container type to its loaded instance.
    """
    partition_path = os.path.join(dirname, partition.value)
    prefixes = get_all_prefixes(partition_path)

    def read_csv(prefix, suffix):
        csv_path = os.path.join(partition_path, prefix + suffix)
        return pd.read_csv(csv_path, parse_dates=True)

    results = {}

    for dtype in dtypes:
        dfs = [read_csv(prefix, dtype.get_csv_suffix()) for prefix in prefixes]
        results[dtype] = dtype(pd.concat(dfs, ignore_index=True))

    return results

def load_dataset(dirname: str, dtypes: list[Type[T]]) -> Dict[Partition, Dict[Type[T], T]]:
    """
    Load all dataset partitions.

    Parameters
    ----------
    dirname : str
        Base dataset directory.
    dtypes : list of DataContainer types
        Containers to load for each split.

    Returns
    -------
    dict
        Mapping of Partition to its loaded containers.
    """
    return {p: load_partition(dirname, p, dtypes) for p in Partition}




