"""
This script contains the code to implement the SVGPR-PS model, using a combination
of GPyTorch and Pyro.
Author: Zach Calhoun
Date: 02/27/2025
"""

import torch
import gpytorch
from gpytorch.models.deep_gps import DeepGPLayer
import pyro

import pyro.distributions.constraints as constraints


class LAYER_1(gpytorch.models.ApproximateGP):
    """
    Define the base class for the preferential sampling model.
    """

    def __init__(
        self,
        num_points,
        area,
        num_inducing=(10, 10),
        inducing_points=None,
        inducing_point_prior=None,
        name_prefix="cox_gp_model",
        learn_inducing_locations=False,
        beta=1.0,
    ):
        # Note -- I adapt some of the code to work with 2D data, so this code is a bit different
        # than the tutorial in that respect.
        self.name_prefix = name_prefix
        self.area = area
        self.beta = beta

        # Define the variational distribution and strategy of the GP
        # We will initialize the inducing points to lie on a grid from 0 to T
        if inducing_points is None:
            inducing_points = self._define_inducing_points(num_inducing)

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=len(inducing_points)
        )

        if inducing_point_prior is not None:
            variational_distribution.initialize_variational_distribution(
                inducing_point_prior
            )

        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )

        super().__init__(variational_strategy)  # , 2, 1)

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def _define_inducing_points(self, num_inducing):
        """
        This function automates setting the inducing points on a grid.
        """
        x = torch.linspace(0, self.area[0], num_inducing[0])
        y = torch.linspace(0, self.area[1], num_inducing[1])
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        X_grid = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
        return X_grid

    def forward(self, x):
        """The standard forward pass for GPs"""
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def guide(self, points, quadrature_points):
        """
        The default guide for the SVGPR-PS model.
        """
        function_distribution = self.pyro_guide(
            torch.vstack([points, quadrature_points]),
            name_prefix=self.name_prefix,
            beta=self.beta,
        )

        with pyro.plate(self.name_prefix + ".times_plate", dim=-1):
            function_samples = pyro.sample(
                self.name_prefix + ".function_samples", function_distribution
            )

        predicted_observed_values, _ = function_samples.split(
            [points.size(0), quadrature_points.size(0)], dim=-1
        )

        return predicted_observed_values

    def model(self, points, quadrature_points):
        """
        A simple model with parameters to learn (and no priors).
        """
        pyro.module(self.name_prefix + ".gp", self)
        function_distribution = self.pyro_model(
            torch.vstack([points, quadrature_points]),
            name_prefix=self.name_prefix,
            beta=self.beta,
        )

        # Draw samples from p(f) at arrival times
        # Also draw samples from p(f) at evenly-spaced points (quadrature_times)
        with pyro.plate(self.name_prefix + ".times_plate", dim=-1):
            function_samples = pyro.sample(
                self.name_prefix + ".function_samples", function_distribution
            )

        # Define the parameters of the model as point values.
        alpha = pyro.param(self.name_prefix + ".alpha", torch.tensor(-1.0))
        beta = pyro.param(self.name_prefix + ".beta", torch.tensor(-1.0))

        ####
        # Convert function samples into intensity samples, using the function above
        # I changed this to multiply the function samples by alpha
        ####
        intensity_samples = (alpha + function_samples).exp()

        # Divide the intensity samples into arrival_intensity_samples and
        # quadrature_intensity_samples
        arrival_intensity_samples, quadrature_intensity_samples = (
            intensity_samples.split([points.size(0), quadrature_points.size(0)], dim=-1)
        )

        ####
        # Compute the log_likelihood, using the method described above
        ####
        arrival_log_intensities = arrival_intensity_samples.log().sum(dim=-1)
        est_num_arrivals = (
            quadrature_intensity_samples.mean(dim=-1) * self.area[0] * self.area[1]
        )
        log_likelihood = arrival_log_intensities - est_num_arrivals
        pyro.factor(self.name_prefix + ".log_likelihood", log_likelihood)

        # Return the predicted observed values at the points
        predicted_observed_values, _ = function_samples.split(
            [points.size(0), quadrature_points.size(0)], dim=-1
        )

        return predicted_observed_values


class LAYER_2(gpytorch.models.PyroGP):
    """
    Define the base class for the preferential sampling model.
    """

    def __init__(
        self,
        num_points,
        inducing_points=None,
        inducing_point_prior=None,
        name_prefix="layer_2_gp_model",
        learn_inducing_locations=True,
    ):
        # Note -- I adapt some of the code to work with 2D data, so this code is a bit different
        # than the tutorial in that respect.
        # self.name_prefix = name_prefix

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=num_points
        )

        inducing_points = torch.randn(num_points, 3)  # .unsqueeze(-1)

        # if inducing_point_prior is not None:
        #     variational_distribution.initialize_variational_distribution(
        #         inducing_point_prior
        #     )
        # else:
        #     inducing_point_prior = gpytorch.distributions.MultivariateNormal(
        #         inducing_points.flatten(),
        #         torch.eye(inducing_points.numel()) * 1.0,
        #     )

        #     variational_distribution.initialize_variational_distribution(
        #         inducing_point_prior
        #     )

        # initialize the points

        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            # noise_constraint=constraints.positive,
            noise_prior=gpytorch.priors.HalfNormalPrior(0.1),
        )

        likelihood.noise = 0.01

        super().__init__(
            variational_strategy, likelihood, num_points, name_prefix=name_prefix
        )

        self.likelihood = likelihood
        self.mean_module = gpytorch.means.ZeroMean()  # (input_size=1)
        self.covar_module = gpytorch.kernels.LinearKernel(
            active_dims=0
        ) + gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(active_dims=(1, 2)))

        # (
        #     gpytorch.kernels.LinearKernel()
        # )  # + gpytorch.kernels.ConstantKernel()

    def forward(self, x):
        """The standard forward pass for GPs"""
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class DeepGP(pyro.nn.PyroModule):
    """
    This class defines a deep GP, in which the first layer is a GP and the second layer is a GP.
    """

    def __init__(
        self,
        X,
        area,
        layer_2_ind_pts=10,
        layer_1_inducing_area=(10.0, 10.0),
        beta=1.0,
        l1_inducing_points=None,
        l1_inducing_point_prior=None,
    ):
        super(DeepGP, self).__init__()
        self.layer1 = LAYER_1(
            X,
            area,
            inducing_points=l1_inducing_points,
            num_inducing=layer_1_inducing_area,
            beta=beta,
            inducing_point_prior=l1_inducing_point_prior,
        )
        self.layer2 = LAYER_2(layer_2_ind_pts)

    def model(self, X, quadrature_points, y):
        """
        The compound model for the deep GP.
        """
        z = self.layer1.model(X, quadrature_points)
        z = torch.cat([z.unsqueeze(-1), X], dim=-1)
        self.layer2.model(z, y)

    def guide(self, X, quadrature_points, y):
        """
        The compound guide for the deep GP.
        """
        z = self.layer1.guide(X, quadrature_points)
        # Concat z and x
        # print(z.shape)
        if len(z.shape) == 1:
            z = z.unsqueeze(-1)
        z = torch.cat([z, X], dim=-1)
        # print(f"z shape: {z.shape}")
        self.layer2.guide(z, y)

    def forward(self, X):
        """
        From the first layer, pass the mean of the GP to the second layer.
        """
        # Sample from the model
        z = self.layer1(X)
        z = torch.cat([z.mean.unsqueeze(-1), X], dim=-1)
        return self.layer2(z)
