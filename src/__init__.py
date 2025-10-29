"""
Set up the package components
"""

from .models import TempModel, DewpointModel
from .utils import SimpleLogger, init_inducing_points, set_up_loss
from .trainers import train_model, validate_model, generate_maps
