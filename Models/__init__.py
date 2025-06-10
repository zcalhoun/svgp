"""
This script takes care of loading the models and likelihoods.
"""

from gpytorch.likelihoods import GaussianLikelihood, StudentTLikelihood
from .temperature import TempModel


def load_model(variable, inducing_points):
    """
    Loads the model based on the variable and inducing points.
    """
    if variable == "tempAvg":
        return TempModel(inducing_points)
    else:
        raise ValueError(f"Unsupported variable: {variable}. Please use 'tempAvg'.")


def load_likelihood(likelihood):
    """
    Set up the likelihood for the model
    """
    if likelihood == "Gaussian":
        return GaussianLikelihood()
    elif likelihood == "Student":
        return StudentTLikelihood()
    else:
        raise ValueError(f"Unknown likelihood: {likelihood}")
