from .datasets import *
from .unc import UNC_Dataset


def load(dataset, file_path):
    """
    This function provides an interface for loading the datasets.
    """
    if dataset == "UNC":
        return UNC_Dataset(file_path)
    else:
        raise ValueError("The dataset must be one of 'UNC'.")
