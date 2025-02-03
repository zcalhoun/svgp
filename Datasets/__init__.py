# from .datasets import *
from .unc import UNC_Dataset


def load(dataset, file_path):
    """
    This function provides an interface for loading the datasets.
    """
    if dataset == "UNC":
        return UNC_Dataset(file_path)

    if dataset == "UNC_periodic":
        return UNC_Dataset(file_path, periodic_features=True)

    if dataset == "UNC_periodic_time":
        return UNC_Dataset(file_path, periodic_features=True, spatial_features=False)

    if dataset == "UNC_periodic_covariates":
        return UNC_Dataset(file_path, periodic_features=True, covariates=True)

    raise ValueError("The dataset must be one of 'UNC'.")
