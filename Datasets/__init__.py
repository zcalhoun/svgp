# from .datasets import *
from .unc import UNC_Dataset
from .wu_mini import WeatherUnderground


def load(dataset, file_path, **kwargs):
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

    if dataset == "WU_mini":
        return WeatherUnderground(file_path, **kwargs)

    raise ValueError("The dataset must be one of 'UNC'.")
