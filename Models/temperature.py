"""
This file contains the temperature model.

"""

import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)


class TempModel(ApproximateGP):
    """
    The Temperature model based on an SVGP. This model assumes
    the following covariates:

    ["t2m", "PC1", "lat", "lon", "sin_hour", "cos_hour", "hour"]
    """

    def __init__(self, inducing_points):
        """
        Define the VNNGP model
        """

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0)
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)

        mean_weights = torch.ones(2)
        mean_weights[1] = 0.1
        self.mean_module = gpytorch.means.LinearMean(2)
        self.mean_module.weights.data = mean_weights

        # ["t2m", "PC1", "lat", "lon", "sin_hour", "cos_hour", "hour"]
        self.covar_module = gpytorch.kernels.ScaleKernel(
            (
                gpytorch.kernels.MaternKernel(nu=0.5, active_dims=(4, 5))
                + gpytorch.kernels.ConstantKernel()
            )
            * gpytorch.kernels.MaternKernel(
                nu=1.5, active_dims=(1, 2, 3), ard_num_dims=3
            )
        ) + gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=0.5, active_dims=(2, 3), ard_num_dims=2)
            * gpytorch.kernels.MaternKernel(nu=1.5, active_dims=6)
        )

    def forward(self, x):
        """
        The forward pass of the model.
        """
        if x.dim() == 2:
            mean_x = self.mean_module(x[:, 0:2])
        else:
            mean_x = self.mean_module(x[:, :, 0:2])
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
