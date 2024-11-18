"""
This init function is responsible for loading the correct model based on the model_name argument.
"""

from .vnngp import PeriodicSpatial_VNNGP, BaseVNNGP, BaseVNNGP_PeriodicFeatures


def load(model_name, *args, **kwargs):
    """Handles loading the necessary model."""
    if model_name == "Periodic_VNNGP":
        return PeriodicSpatial_VNNGP(*args, **kwargs)
    if model_name == "BaseVNNGP":
        return BaseVNNGP(*args, **kwargs)
    if model_name == "BaseVNNGP_PeriodicFeatures":
        return BaseVNNGP_PeriodicFeatures(*args, **kwargs)

    raise ValueError(f"Model {model_name} not found")
