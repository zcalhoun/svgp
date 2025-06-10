"""
The purpose of this code is to define a Sparse Variational Gaussian Process (SVGP)
model that can be used to predict the urban heat island effect using
satellite data, and difference loss functions or likelihoods.

I find that a small number of inducing points (1000) is sufficient to model a month's
worth of data, and the model is quickly trained on a GPU (<5 minutes).

"""

import os
import glob
import random

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils import SimpleLogger


def main(args):
    """
    Load the data
    Train the model
    Save the final results!
    """

    logger = SimpleLogger(args.name)
    # Load the data
    logger.info(args)

    # Validate the output directory
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    train_X, train_y, test_X, test_y, unc_X, unc_y, test_df, unc_df, weights = (
        load_data(
            args.data_directory,
            train_size=args.train_size,
            extra_cols=args.extra_cols,
        )
    )

    logger.info(f"Weights shape: {weights.shape}")

    logger.info("Data loaded")
    logger.info(f"Train data shape: {train_X.shape}")

    mean_weights = init_mean_coefs(args.extra_cols)

    logger.start_timer("INIT")

    likelihood = set_up_likelihood(args.likelihood)

    inducing_points = initialize_inducing_points(
        train_X, args.num_inducing_points, method=args.inducing_points_method
    )

    model = set_up_model(
        args.model, inducing_points, mean_weights, extra_cols=args.extra_cols
    )

    # model = SVGP(inducing_points=inducing_points, mean_weights=mean_weights)

    logger.stop_timer("INIT")

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    # Train the model
    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters()},
            {"params": likelihood.parameters()},
        ],
        lr=args.lr,
    )

    # warmup_scheduler = LinearLR(optimizer, start_factor=0.001, total_iters=200)
    t_max = args.num_epochs * (len(train_X) // args.batch_size + 1)
    scheduler = CosineAnnealingLR(optimizer, t_max)
    # scheduler = SequentialLR(
    #     optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[200]
    # )

    mll = set_up_loss(args.loss_function, likelihood, model, train_y.size(0))

    # Create the train dataset
    train_dataset = TensorDataset(train_X, train_y, torch.from_numpy(weights).float())
    # Create the train dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_ds = TensorDataset(test_X, test_y)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    unc_ds = TensorDataset(unc_X, unc_y)
    unc_loader = DataLoader(unc_ds, batch_size=args.batch_size, shuffle=False)

    epoch_losses = []
    best_mse = 1e6
    for epoch in range(args.num_epochs):

        logger.start_timer("TRAIN")
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        train_loss = train(model, likelihood, mll, optimizer, train_loader, scheduler)
        logger.stop_timer("TRAIN")

        logger.start_timer("VALIDATE")
        logger.info("Validating the model")
        val_mae, val_mse, val_nlpd, qce50, qce75, qce95 = validate(
            model, likelihood, test_loader
        )

        unc_mae, unc_mse, unc_nlpd, unc_qce50, unc_qce75, unc_qce95 = validate(
            model, likelihood, unc_loader
        )

        logger.stop_timer("VALIDATE")
        if epoch > 2:
            if val_mse < best_mse:
                best_mse = val_mse
                torch.save(model.state_dict(), os.path.join(args.output, "model.pt"))
                torch.save(
                    likelihood.state_dict(), os.path.join(args.output, "likelihood.pt")
                )

        logger.info(
            f"Epoch {epoch + 1}/{args.num_epochs} - "
            f"Train Loss: {train_loss:.3f} - "
            f"Validation MAE: {val_mae:.3f} - "
            f"Validation NLPD: {val_nlpd:.3f} - "
            f"UNC MAE: {unc_mae:.3f} - "
            f"UNC NLPD: {unc_nlpd:.3f} - "
        )
        epoch_losses.append(
            [
                epoch,
                train_loss,
                val_mae,
                val_mse,
                val_nlpd,
                qce50,
                qce75,
                qce95,
                unc_mae,
                unc_mse,
                unc_nlpd,
                unc_qce50,
                unc_qce75,
                unc_qce95,
            ]
        )

    # Load the state dict
    logger.info("Loading the best model")
    model.load_state_dict(torch.load(os.path.join(args.output, "model.pt")))
    likelihood.load_state_dict(torch.load(os.path.join(args.output, "likelihood.pt")))

    # After training, let's get predictions on all of the data
    preds, variances = predict(model, likelihood, test_loader)
    test_df["preds"] = preds
    test_df["variances"] = variances

    preds, variances = predict(model, likelihood, unc_loader)
    unc_df["preds"] = preds
    unc_df["variances"] = variances

    # Save the predictions to a CSV file
    test_df.to_csv(os.path.join(args.output, "val_predictions.csv"), index=False)
    unc_df.to_csv(os.path.join(args.output, "unc_predictions.csv"), index=False)

    epoch_losses = pd.DataFrame(
        epoch_losses,
        columns=[
            "epoch",
            "train_loss",
            "val_mae",
            "val_mse",
            "val_nlpd",
            "qce50",
            "qce75",
            "qce95",
            "unc_mae",
            "unc_mse",
            "unc_nlpd",
            "unc_qce50",
            "unc_qce75",
            "unc_qce95",
        ],
    )
    epoch_losses.to_csv(os.path.join(args.output, "epoch_losses.csv"), index=False)


def set_up_model(model, inducing_points, mean_weights=None, extra_cols=None):
    """
    Set up the model for training
    """

    if model == "base":
        return SVGP_BASE(inducing_points=inducing_points, mean_weights=mean_weights)
    elif model == "periodic":
        return SVGP(
            inducing_points=inducing_points,
            mean_weights=mean_weights,
            extra_cols=extra_cols,
        )
    else:
        raise ValueError(f"Unknown model: {model}")


class IWPLL(gpytorch.mlls.PredictiveLogLikelihood):
    """
    Importance Weighted Predictive Log Likelihood (IWPLL) for GPyTorch.
    This class extends the PredictiveLogLikelihood to support importance weighting.
    """

    def _log_likelihood_term(self, approximate_dist_f, target, weights=None, **kwargs):
        if weights is not None:
            return self.likelihood.log_marginal(target, approximate_dist_f) @ weights
        else:
            return self.likelihood.log_marginal(
                target, approximate_dist_f, **kwargs
            )  # @ weights


def set_up_loss(loss_function, likelihood, model, size):
    """
    Set up the loss function for the model
    """

    if loss_function == "ELBO":
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=size)
    elif loss_function == "PLL":
        mll = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, num_data=size)
    elif loss_function == "IW-PLL":
        mll = IWPLL(likelihood, model, num_data=size)
    else:
        raise ValueError(f"Unknown loss function: {loss_function}")

    return mll


def predict(model, likelihood, test_loader):
    """
    Predict the model
    """

    model.eval()
    likelihood.eval()
    preds = []
    pred_vars = []
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(100):

        for X, _ in test_loader:
            if torch.cuda.is_available():
                X = X.cuda()

            pred = likelihood(model(X))
            pred_mean = pred.mean
            if pred_mean.dim() == 1:
                preds.append(pred_mean.cpu().numpy())
                pred_vars.append(pred.variance.cpu().numpy())
            else:
                # Handle Student T outputs
                preds.append(pred_mean.mean(dim=0).cpu().numpy())
                pred_vars.append(pred.sample().var(axis=0).cpu().numpy())

    return np.concatenate(preds), np.concatenate(pred_vars)


def validate(model, likelihood, test_loader):
    """
    Validate the model
    """

    model.eval()
    likelihood.eval()
    mae = 0
    mse = 0
    nlpd = 0
    qce50 = 0
    qce75 = 0
    qce95 = 0
    count = 0
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(1000):

        for X, y in test_loader:
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()

            preds = likelihood(model(X))

            mean_preds = preds.mean
            if mean_preds.dim() == 1:
                mae += torch.sum(torch.abs(mean_preds - y))
                mse += torch.sum((mean_preds - y) ** 2)
                nlpd += torch.sum(-preds.log_prob(y))

                qce50 += quantile_coverage_error(preds, y, 50.0).item() * y.shape[0]

                qce75 += quantile_coverage_error(preds, y, 75.0).item() * y.shape[0]

                qce95 += quantile_coverage_error(preds, y, 95.0).item() * y.shape[0]
            else:
                mae += torch.sum(torch.abs(mean_preds.mean(axis=0) - y))
                mse += torch.sum((mean_preds.mean(axis=0) - y) ** 2)
                S, _ = mean_preds.shape
                nlpd += torch.sum(
                    -torch.logsumexp(preds.log_prob(y), dim=0) + np.log(S)
                )
                samples = preds.sample()
                qce50 += t_qce_coverage(samples, y, alpha=50.0) * y.shape[0]
                qce75 += t_qce_coverage(samples, y, alpha=75.0) * y.shape[0]
                qce95 += t_qce_coverage(samples, y, alpha=95.0) * y.shape[0]
            count += y.size(0)
    mae /= count
    mse /= count
    nlpd /= count
    qce50 /= count
    qce75 /= count
    qce95 /= count

    return mae.item(), mse.item(), nlpd.item(), qce50, qce75, qce95


def train(model, likelihood, mll, optimizer, train_loader, scheduler):
    """
    This function runs the train loader to train the model.
    """

    model.train()
    likelihood.train()

    epoch_loss = 0
    epoch_count = 0

    for x_batch, y_batch, w in train_loader:
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
        w = w.cuda()
        optimizer.zero_grad()
        output = model(x_batch)
        loss = -mll(output, y_batch, weights=w)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_count += y_batch.size(0)
        scheduler.step()

    total_loss = epoch_loss / epoch_count

    return total_loss


def initialize_inducing_points(train_X, num_inducing_points, method="random"):
    """
    This function initializes the inducing points for the model.

    There are two supported methods at this time:
    1. Randomly sample inducing points completely.
    2. Randomly sample inducing points from the training data.
    """

    # For reproducibility...
    torch.manual_seed(5)

    if method == "random":
        inducing_points = torch.rand(num_inducing_points, train_X.shape[1])
        inducing_points = (
            inducing_points * (train_X.max(0).values - train_X.min(0).values)
            + train_X.min(0).values
        )
    elif method == "random_train":
        M = 20000
        inducing_points = train_X[torch.randperm(M)][:num_inducing_points]
    else:
        raise NotImplementedError(
            f"Method {method} not implemented for initializing inducing points."
        )

    return inducing_points


def init_mean_coefs(extra_cols):
    """
    This function calculates the mean coefficients and the bias for the model.

    This will drastically improve initial model performance.
    """
    if extra_cols is not None:
        mean_weights = torch.ones(1 + len(extra_cols))
        for i, col in enumerate(extra_cols):
            if col == "evi":
                mean_weights[i + 1] = -0.1
            else:
                mean_weights[i + 1] = 0.1
    else:
        mean_weights = torch.ones(1)

    return mean_weights


class SVGP(ApproximateGP):
    """
    The SVGP model with a periodic kernel added.

    """

    def __init__(self, inducing_points, mean_weights=None, extra_cols=None):
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

        self.mean_module = gpytorch.means.LinearMean(1 + len(extra_cols))
        self.mean_module.weights.data = mean_weights
        self.mean_module.bias.data = torch.zeros(1)

        # hour_dim = 1 + len(extra_cols)
        # covar_dim = [1 + i for i in range(len(extra_cols))]
        # coord_dim = [2 + len(extra_cols), 3 + len(extra_cols)]
        # ['t2m', 'evi', 'lat', 'lon', 'sin_hour', 'cos_hour','hour']
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

        # self.covar_module.kernels[0].base_kernel.kernels[0].kernels[
        #     0
        # ].period_length = 24.0

        self.terms = 1 + len(extra_cols)

    def forward(self, x):
        # print(x.shape)
        if x.dim() == 2:
            mean_x = self.mean_module(x[:, 0 : self.terms])
        else:
            mean_x = self.mean_module(x[:, :, 0 : self.terms])
        # print("OK")
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SVGP_BASE(ApproximateGP):
    """
    The base SVGP model used for training.

    """

    def __init__(self, inducing_points, mean_weights=None):
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

        self.mean_module = gpytorch.means.LinearMean(1)
        self.mean_module.weights.data = mean_weights
        self.mean_module.bias.data = torch.zeros(1)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=0.5, active_dims=(2, 3), ard_num_dims=2)
            * gpytorch.kernels.MaternKernel(nu=1.5, active_dims=1)
        )

    def forward(self, x):
        # print(x.shape)
        if x.dim() == 2:
            mean_x = self.mean_module(x[:, 0])
        else:
            mean_x = self.mean_module(x[:, :, 0])
        # print("OK")
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def load_data(data_directory, train_size=0.8, random_seed=42, extra_cols=None):
    """
    This function does a lot...it:
    1. Loads the data using the provided data directory.
    2. The load is split into training and test sets based on the list of stations.
    3. Features are extracted from the data, such as the periodized hour of the day.
    3. The data is then normalized to be fed into the model.
    """

    random.seed(random_seed)
    wu_path = os.path.join(data_directory, "train")
    all_stations = glob.glob(os.path.join(wu_path, "station=*"))
    random.shuffle(all_stations)

    split_index = int(len(all_stations) * train_size)
    train_stations = all_stations[:split_index]
    test_stations = all_stations[split_index:]

    train_paths = get_all_paths(train_stations)
    test_paths = get_all_paths(test_stations)

    train_df = load_dataset(train_paths)
    test_df = load_dataset(test_paths)
    unc_df = pd.read_csv(os.path.join(data_directory, "unc_test.csv"))

    train_df["t_s_avg"] = (train_df["day_t_s"] + train_df["night_t_s"]) / 2
    test_df["t_s_avg"] = (test_df["day_t_s"] + test_df["night_t_s"]) / 2
    unc_df["t_s_avg"] = (unc_df["day_t_s"] + unc_df["night_t_s"]) / 2

    min_date = train_df["date"].min()
    train_df["hour"] = train_df["date"] - min_date
    test_df["hour"] = test_df["date"] - min_date
    unc_df["hour"] = pd.to_datetime(unc_df["time"]) - min_date

    train_df["hour"] = train_df["hour"].dt.total_seconds() / 3600
    test_df["hour"] = test_df["hour"].dt.total_seconds() / 3600
    unc_df["hour"] = unc_df["hour"].dt.total_seconds() / 3600

    periodize(train_df)
    periodize(test_df)
    periodize(unc_df)

    cols = ["t2m"]
    if extra_cols is not None:
        cols.extend(extra_cols)

    # weights = initialize_weights(train_df[extra_cols].values)
    weights = initialize_weights(train_df[["lon", "lat"]].values)
    cols.extend(["lat", "lon", "sin_hour", "cos_hour", "hour"])

    train_X = train_df[cols].values
    train_y = train_df["tempAvg"].values

    test_X = test_df[cols].values
    test_y = test_df["tempAvg"].values

    unc_df.rename(columns={"Lat": "lat", "Lon": "lon"}, inplace=True)
    unc_X = unc_df[cols].values
    unc_y = unc_df["Temperature"].values

    assert train_X.shape[-1] == test_X.shape[-1]

    train_X = torch.tensor(train_X, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    test_X = torch.tensor(test_X, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32)
    unc_X = torch.tensor(unc_X, dtype=torch.float32)
    unc_y = torch.tensor(unc_y, dtype=torch.float32)

    if extra_cols is not None:
        mu = torch.mean(train_X[:, 1 : 1 + len(extra_cols)], dim=0)
        std = torch.std(train_X[:, 1 : 1 + len(extra_cols)], dim=0)

        train_X[:, 1 : 1 + len(extra_cols)] = (
            train_X[:, 1 : 1 + len(extra_cols)] - mu
        ) / std
        test_X[:, 1 : 1 + len(extra_cols)] = (
            test_X[:, 1 : 1 + len(extra_cols)] - mu
        ) / std
        unc_X[:, 1 : 1 + len(extra_cols)] = (
            unc_X[:, 1 : 1 + len(extra_cols)] - mu
        ) / std

    return train_X, train_y, test_X, test_y, unc_X, unc_y, test_df, unc_df, weights


def periodize(df, period=24):
    """This creates the features needed to periodize the hour of the day."""
    df["sin_hour"] = np.sin(2 * df["hour"] * np.pi / period)
    df["cos_hour"] = np.cos(2 * df["hour"] * np.pi / period)


def initialize_weights(columns):
    """
    Initialize the weights for the extra columns.
    This is used to improve the initial performance of the model.
    """
    features = np.unique(columns, axis=0)

    # mean = features.mean(axis=0)
    # std = features.std(axis=0)
    # features = (features - mean) / std

    # columns = (columns - mean) / std

    kde = KernelDensity(kernel="exponential", bandwidth=0.1).fit(features)

    w = kde.score_samples(columns)
    w = 1 / np.exp(w)
    w = w / np.sum(w) * len(w)
    w[w > 1] = 1  # Cap the weights to avoid extreme values
    w = w / np.sum(w) * len(w)  # Normalize the weights
    return w


def load_dataset(paths):
    """
    This just opens up the parquet files and loads them into a single dataframe.
    """
    dfs = [pd.read_parquet(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)


def get_all_paths(station_path_list, month="*", year="*"):
    """
    This function finds all of the paths to the data files for
    a given set of stations, month, and year.
    """

    all_paths = []
    for station_path in station_path_list:
        all_paths.extend(glob.glob(os.path.join(station_path, f"*.parquet")))
    return all_paths


def quantile_coverage_error(pred_dist, test_y, quantile):
    """
    Quantile coverage error for normal distributions
    """
    if quantile <= 0 or quantile >= 100:
        raise NotImplementedError("Quantile must be between 0 and 100")
    # combine_dim = -2 if isinstance(pred_dist, MultitaskMultivariateNormal) else -1
    standard_normal = torch.distributions.Normal(loc=0.0, scale=1.0)
    deviation = standard_normal.icdf(torch.as_tensor(0.5 + 0.5 * (quantile / 100)))
    lower = pred_dist.mean - deviation * pred_dist.stddev
    upper = pred_dist.mean + deviation * pred_dist.stddev
    n_samples_within_bounds = ((test_y > lower) * (test_y < upper)).sum(-1)
    fraction = n_samples_within_bounds / test_y.shape[-1]
    return fraction - quantile / 100


def t_qce_coverage(y_samples, y_true, alpha=95.0):
    """
    Compute empirical coverage of central prediction intervals using PyTorch.

    Parameters
    ----------
    y_samples : torch.Tensor
        Shape (S, N) — predictive samples from the posterior
    y_true : torch.Tensor
        Shape (N,) — true labels
    alpha : float
        Desired coverage level (e.g., 90. for 90%)

    Returns
    -------
    coverage error: float
        Distance from the desired coverage level
    """
    # Compute quantiles across samples (dim=0 → across S samples per point)
    lower = torch.quantile(y_samples, q=(1 - alpha / 100) / 2, dim=0)
    upper = torch.quantile(y_samples, q=(1 + alpha / 100) / 2, dim=0)

    # Check if true values are inside the intervals
    inside = (y_true >= lower) & (y_true <= upper)

    fraction = inside.float().mean()

    return (fraction - alpha / 100).item()

    # return inside.float().mean().item()


def set_up_likelihood(likelihood):
    """
    Set up the likelihood for the model
    """

    if likelihood == "Gaussian":
        return gpytorch.likelihoods.GaussianLikelihood()
    elif likelihood == "Student":
        return gpytorch.likelihoods.StudentTLikelihood()
    else:
        raise ValueError(f"Unknown likelihood: {likelihood}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VNNGP model")
    parser.add_argument(
        "--data_directory", type=str, default="data.csv", help="Path to the data file"
    )
    parser.add_argument(
        "--output", type=str, default="output.csv", help="Path to the output file"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["base", "periodic"],
        help="Model to use for training",
    )
    # Data parameters
    parser.add_argument(
        "--year", type=str, default="*", help="Year to use for training"
    )
    parser.add_argument(
        "--month", type=str, default="*", help="Month to use for training"
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.8,
        help="Proportion of data to use for training",
    )

    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of epochs to train"
    )

    parser.add_argument(
        "--name", type=str, default="VNNGP", help="Name of the experiment"
    )

    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate for the optimizer"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training and testing",
    )

    parser.add_argument(
        "--inducing_points_method",
        type=str,
        default="random_train",
        choices=["random", "random_train"],
        help="Method to use for initializing inducing points",
    )

    # Add a loss function with choices "ELBO" and "PLL"
    parser.add_argument(
        "--loss_function",
        type=str,
        default="ELBO",
        choices=["ELBO", "PLL", "IW-PLL"],
        help="Loss function to use",
    )

    parser.add_argument(
        "--likelihood",
        type=str,
        default="Gaussian",
        choices=["Gaussian", "Student"],
        help="Likelihood to use",
    )

    parser.add_argument(
        "--num_inducing_points",
        type=int,
        default=100,
        help="Number of inducing points to use for the model",
    )

    parser.add_argument(
        "--variational_lr",
        type=float,
        default=0.1,
        help="Learning rate for the variational optimizer",
    )

    parser.add_argument(
        "--extra_cols",
        type=str,
        nargs="*",
        default=None,
        help="Extra columns to use for training",
    )

    args = parser.parse_args()
    main(args)
