import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational.nearest_neighbor_variational_strategy import (
    NNVariationalStrategy,
)


class PeriodicSpatial_VNNGP(ApproximateGP):
    """This class defines the model for the VNN-GP, where the kernel is defined
    with a temporal dimension as well as a periodic spatial dimension."""

    def __init__(self, inducing_points, likelihood, k=256, training_batch_size=256):

        m, _ = inducing_points.shape
        self.m = m
        self.k = k

        variational_distribution = (
            gpytorch.variational.MeanFieldVariationalDistribution(m)
        )

        if torch.cuda.is_available():
            inducing_points = inducing_points.cuda()

        variational_strategy = NNVariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            k=k,
            training_batch_size=training_batch_size,
        )
        super(PeriodicSpatial_VNNGP, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ZeroMean()

        # Mean temperature kernel
        k1 = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=0.5, active_dims=0)
            + gpytorch.kernels.PeriodicKernel(active_dims=0)
        )

        k2 = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=0.5, active_dims=(1, 2))
            * (
                gpytorch.kernels.PeriodicKernel(active_dims=0)
                + gpytorch.kernels.ConstantKernel(active_dims=0)
            )
        )

        self.covar_module = k1 + k2
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, prior=False, **kwargs):
        if x is not None:
            if x.dim() == 1:
                x = x.unsqueeze(-1)
        return self.variational_strategy(x=x, prior=False, **kwargs)
