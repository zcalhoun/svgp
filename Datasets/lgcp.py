"""
This code handles creating the Log Gaussian Cox Process (LGCP) so we can properly weight
the data that is fed into the model. A lot of this code was adapted from the code
provided by the GPytorch team for the LGCP tutorial:
https://docs.gpytorch.ai/en/v1.13/examples/07_Pyro_Integration/Cox_Process_Example.html


"""

import gc

import torch
import gpytorch
import pyro


def lgcp_weight(
    train_coords,
    test_coords,
    num_ip_dim=3,
    num_qp_dim=40,
    num_iter=400,
    num_particles=32,
):
    """
    train_coords: The coordinates of the training data.
    test_coords: The coordinates of the test data.
    num_ip_dim: The number of inducing points to use in each dimension.
    """

    vmin = train_coords.min(dim=0).values
    vmax = train_coords.max(dim=0).values

    X_train = (train_coords - vmin) / (vmax - vmin)
    X_test = (test_coords - vmin) / (vmax - vmin)

    inducing_points = create_inducing_points(X_train, num_ip_dim=num_ip_dim)

    model = GPModel(len(X_train), inducing_points)

    quad_points = create_inducing_points(X_train, num_ip_dim=num_qp_dim)

    pyro.clear_param_store()

    train(model, X_train, quad_points, num_iter=num_iter, num_particles=num_particles)

    model.eval()
    with torch.no_grad():
        function_dist = model(X_train)
        intensity_samples = function_dist(torch.Size([1000])).exp() * model.num_obs
        _, mean_train, _ = percentiles_from_samples(intensity_samples)

        function_dist = model(X_test)
        intensity_samples = function_dist(torch.Size([1000])).exp() * model.num_obs
        _, mean_test, _ = percentiles_from_samples(intensity_samples)

    train_weights = calc_weights(mean_train)
    test_weights = calc_weights(mean_test)

    del model, function_dist, intensity_samples
    gc.collect()
    torch.cuda.empty_cache()

    return train_weights, test_weights


def calc_weights(intensity):
    """
    Calculate the inverse weights, then normalize them so they sum to the number of samples.
    """
    w = 1 / intensity
    w = w / w.sum() * w.numel()

    return w.float()


def percentiles_from_samples(samples, percentiles=[0.01, 0.5, 0.95]):
    """
    Calculates the percentile from the samples to get the mean.
    """
    num_samples = samples.size(0)
    samples = samples.sort(dim=0)[0]

    # Get samples corresponding to percentile
    percentile_samples = [
        samples[int(num_samples * percentile)] for percentile in percentiles
    ]

    return percentile_samples


def train(model, X, qp, lr=0.01, num_iter=200, num_particles=32):
    """
    Take care of the training loop.
    """
    optimizer = pyro.optim.Adam({"lr": lr})
    loss = pyro.infer.Trace_ELBO(
        num_particles=num_particles, vectorize_particles=True, retain_graph=True
    )
    infer = pyro.infer.SVI(model.model, model.guide, optimizer, loss=loss)

    model.train()
    # loader = tqdm.notebook.tqdm(range(num_iter))
    for i in range(num_iter):
        loss = infer.step(X, qp)


def create_inducing_points(X, num_ip_dim=5):
    """
    Simple helper function to create a grid of inducing points.

    """
    xx = torch.linspace(X[:, 0].min(), X[:, 0].max(), num_ip_dim)
    yy = torch.linspace(X[:, 1].min(), X[:, 1].max(), num_ip_dim)

    X1, X2 = torch.meshgrid(xx, yy)

    X = torch.column_stack([X1.flatten(), X2.flatten()])

    return X


class GPModel(gpytorch.models.ApproximateGP):
    """
    Log Gaussian Cox Process object

    """

    def __init__(self, num_obs, inducing_points, name_prefix="cox_gp_model"):
        self.name_prefix = name_prefix
        self.num_obs = num_obs

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0),
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution
        )

        # Define model
        super().__init__(variational_strategy=variational_strategy)

        # Define mean and kernel
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5)
        )

    def forward(self, times):
        mean = self.mean_module(times)
        covar = self.covar_module(times)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def guide(self, arrival_times, quadrature_times):
        function_distribution = self.pyro_guide(
            torch.vstack([arrival_times, quadrature_times])
        )

        # Draw samples from q(f) at arrival_times
        # Also draw samples from q(f) at evenly-spaced points (quadrature_times)
        with pyro.plate(self.name_prefix + ".times_plate", dim=-1):
            pyro.sample(self.name_prefix + ".function_samples", function_distribution)

    def model(self, arrival_times, quadrature_times):
        pyro.module(self.name_prefix + ".gp", self)
        function_distribution = self.pyro_model(
            torch.vstack([arrival_times, quadrature_times])
        )

        # Draw samples from p(f) at arrival times
        # Also draw samples from p(f) at evenly-spaced points (quadrature_times)
        with pyro.plate(self.name_prefix + ".times_plate", dim=-1):
            function_samples = pyro.sample(
                self.name_prefix + ".function_samples", function_distribution
            )

        ####
        # Convert function samples into intensity samples, using the function above
        ####
        intensity_samples = function_samples.exp() * self.num_obs

        # Divide the intensity samples into arrival_intensity_samples and quadrature_intensity_samples
        arrival_intensity_samples, quadrature_intensity_samples = (
            intensity_samples.split(
                [arrival_times.size(0), quadrature_times.size(0)], dim=-1
            )
        )

        ####
        # Compute the log_likelihood, using the method described above
        ####
        arrival_log_intensities = arrival_intensity_samples.log().sum(dim=-1)
        # print(quadrature_intensity_samples.size())
        est_num_arrivals = quadrature_intensity_samples.mean(
            dim=-1
        )  # .mul(self.max_time)
        log_likelihood = arrival_log_intensities - est_num_arrivals
        pyro.factor(self.name_prefix + ".log_likelihood", log_likelihood)
