"""
This file contains all of the models used in my experiments.

Classes:
    * VNNGP - the base model class.
    * BaseVNNGP - a simple VNN-GP model with a Matern kernel.
    * PeriodicSpatial_VNNGP - a VNN-GP model with a Matern kernel and a periodic spatial kernel.

"""

import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational.nearest_neighbor_variational_strategy import (
    NNVariationalStrategy,
)


class VNNGP(ApproximateGP):
    """The base VNNGP model class."""

    def __init__(self, inducing_points, likelihood, k=256, training_batch_size=256):
        m, _ = inducing_points.shape
        self.m = m
        self.k = k

        variational_distribution = (
            gpytorch.variational.MeanFieldVariationalDistribution(m)
        )

        variational_strategy = NNVariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            k=k,
            training_batch_size=training_batch_size,
        )

        super(VNNGP, self).__init__(variational_strategy)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, prior=False, **kwargs):
        if x is not None:
            if x.dim() == 1:
                x = x.unsqueeze(-1)
        return self.variational_strategy(x=x, prior=False, **kwargs)


class BaseVNNGP(VNNGP):
    """This class defines a simple VNN-GP model with a Matern kernel."""

    def __init__(self, inducing_points, likelihood, k=256, training_batch_size=256):

        super(BaseVNNGP, self).__init__(
            inducing_points, likelihood, k=k, training_batch_size=training_batch_size
        )

        self.mean_module = gpytorch.means.ZeroMean()

        self.covar_module = (
            gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=0.5, active_dims=0)
                + gpytorch.kernels.PeriodicKernel(active_dims=0)
            )
            + gpytorch.kernels.ConstantKernel()
        )

        self.likelihood = likelihood


class BaseVNNGP_PeriodicFeatures(VNNGP):
    """This class defines a simple VNN-GP model with a Matern kernel."""

    def __init__(self, inducing_points, likelihood, k=256, training_batch_size=256):

        super(BaseVNNGP_PeriodicFeatures, self).__init__(
            inducing_points, likelihood, k=k, training_batch_size=training_batch_size
        )

        self.mean_module = gpytorch.means.ConstantMean()

        self.covar_module = (
            gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=0.5, active_dims=0)
            )
            + gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=0.5, active_dims=(3, 4))
            )
            + gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=1.5, active_dims=(1, 2), ard=2)
                * (
                    gpytorch.kernels.MaternKernel(nu=0.5, active_dims=(3, 4))
                    + gpytorch.kernels.ConstantKernel()
                )
            )
            + gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=0.5, active_dims=(1, 2), ard=2)
                * gpytorch.kernels.MaternKernel(nu=1.5, active_dims=0)
            )
        ) + gpytorch.kernels.ConstantKernel()

        self.likelihood = likelihood


class PeriodicSpatial_VNNGP(VNNGP):
    """This class defines the model for the VNN-GP, where the kernel is defined
    with a temporal dimension as well as a periodic spatial dimension."""

    def __init__(
        self, inducing_points, likelihood, k=256, training_batch_size=256, period=1.0
    ):

        super(PeriodicSpatial_VNNGP, self).__init__(
            inducing_points, likelihood, k=k, training_batch_size=training_batch_size
        )

        self.mean_module = gpytorch.means.ZeroMean()

        period_prior = gpytorch.priors.NormalPrior(period, 1e-6)

        # Mean temperature kernel
        k1 = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=0.5, active_dims=0)
            + gpytorch.kernels.PeriodicKernel(
                period_length_prior=period_prior, active_dims=0
            )
        )

        k2 = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=0.5, active_dims=(1, 2))
            * gpytorch.kernels.PeriodicKernel(
                period_length_prior=period_prior, active_dims=0
            )
        )

        self.covar_module = k1 + k2

        self.covar_module.kernels[0].base_kernel.kernels[1].period_length = period
        self.covar_module.kernels[1].base_kernel.kernels[1].period_length = period
        self.covar_module.kernels[1].base_kernel.kernels[1].length_scale = 0.01

        self.likelihood = likelihood
