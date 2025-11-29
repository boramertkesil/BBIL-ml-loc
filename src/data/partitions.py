from enum import Enum

class Partition(Enum):
    """
    Enumeration for dataset partitions.
    """
    TRAIN = "train"
    TEST  = "test"
    VALID = "valid"

TRAIN = Partition.TRAIN
TEST  = Partition.TEST
VALID = Partition.VALID