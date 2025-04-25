"""
The purpose of this code is to define a Variational Nearest
Neighbor Gaussian Process (VNNGP) model, and to experiment with
using this model on the complete Weather Underground dataset.

Ideally, the purpose of this experiment is to determine whether
the VNNGP can provide us with reasonably good estimates of neighborhood
urban heat island effect.

"""

import os
import glob
import random

import numpy as np
import pandas as pd

# Import this to confirm this package is installed!
# This is a key dependency for the VNNGP, and if not
# installed, the code falls back to a much slower
# implementation of the VNNGP.
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)
import torch
from torch.utils.data import DataLoader, TensorDataset
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

    train_X, train_y, test_X, test_y, test_df = load_data(
        args.data_directory,
        year=args.year,
        month=args.month,
        train_size=args.train_size,
    )

    logger.info("Data loaded")
    logger.info(f"Train data shape: {train_X.shape}")

    mean_weights = init_mean_coefs(train_y)

    logger.start_timer("INIT")

    likelihood = set_up_likelihood(args.likelihood)

    inducing_points = torch.rand(args.num_inducing_points, 7)
    inducing_points = (
        inducing_points * (train_X.max(0).values - train_X.min(0).values)
        + train_X.min(0).values
    )
    model = SVGP(inducing_points=inducing_points, mean_weights=mean_weights)

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

    mll = set_up_loss(args.loss_function, likelihood, model, train_y.size(0))

    # Create the train dataset
    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    # Create the train dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    test_ds = TensorDataset(test_X, test_y)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    epoch_losses = []
    best_mse = 1e6
    for epoch in range(args.num_epochs):

        logger.start_timer("TRAIN")
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        train_loss = train(model, likelihood, mll, optimizer, train_loader)
        logger.stop_timer("TRAIN")

        logger.start_timer("VALIDATE")
        logger.info("Validating the model")
        val_mae, val_mse, val_nlpd, val_qce = validate(model, likelihood, test_loader)
        logger.stop_timer("VALIDATE")
        if epoch > 10:
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
            f"Validation MSE: {val_mse:.3f} - "
            f"Validation NLPD: {val_nlpd:.3f} - "
            f"Validation QCE: {val_qce:.3f}"
        )
        epoch_losses.append([epoch, train_loss, val_mae, val_mse, val_nlpd, val_qce])

    # Load the state dict
    logger.info("Loading the best model")
    model.load_state_dict(torch.load(os.path.join(args.output, "model.pt")))
    likelihood.load_state_dict(torch.load(os.path.join(args.output, "likelihood.pt")))

    # After training, let's get predictions on all of the data
    preds = predict(model, likelihood, test_loader)
    test_df["preds"] = preds

    # Save the predictions to a CSV file
    test_df.to_csv(os.path.join(args.output, "predictions.csv"), index=False)

    epoch_losses = pd.DataFrame(
        epoch_losses,
        columns=["epoch", "train_loss", "val_mae", "val_mse", "val_nlpd", "val_qce"],
    )
    epoch_losses.to_csv(os.path.join(args.output, "epoch_losses.csv"), index=False)


def set_up_loss(loss_function, likelihood, model, size):
    """
    Set up the loss function for the model
    """

    if loss_function == "ELBO":
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=size)
    elif loss_function == "PLL":
        mll = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, num_data=size)
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
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(100):

        for X, _ in test_loader:
            if torch.cuda.is_available():
                X = X.cuda()

            pred = likelihood(model(X))
            pred_mean = pred.mean
            if pred_mean.dim() == 1:
                preds.append(pred_mean.cpu().numpy())
            else:
                # Handle Student T outputs
                preds.append(pred_mean.mean(dim=0).cpu().numpy())

    return np.concatenate(preds)


def validate(model, likelihood, test_loader):
    """
    Validate the model
    """

    model.eval()
    likelihood.eval()
    mae = 0
    mse = 0
    nlpd = 0
    qce = 0
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
                qce += (
                    gpytorch.metrics.quantile_coverage_error(preds, y, 0.95).item()
                    * y.shape[0]
                )
            else:
                mae += torch.sum(torch.abs(mean_preds.mean(axis=0) - y))
                mse += torch.sum((mean_preds.mean(axis=0) - y) ** 2)
                S, _ = mean_preds.shape
                nlpd += torch.sum(
                    -torch.logsumexp(preds.log_prob(y), dim=0) + np.log(S)
                )
                samples = preds.sample()
                qce += qce_coverage(samples, y, alpha=0.95) * y.shape[0]
            count += y.size(0)
    mae /= count
    mse /= count
    nlpd /= count
    qce /= count

    return mae.item(), mse.item(), nlpd.item(), qce


def train(model, likelihood, mll, optimizer, train_loader):
    """
    This function runs the train loader to train the model.
    """

    model.train()
    likelihood.train()

    epoch_loss = 0
    epoch_count = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
        optimizer.zero_grad()
        output = model(x_batch)
        loss = -mll(output, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_count += y_batch.size(0)

    total_loss = epoch_loss / epoch_count

    return total_loss


def init_mean_coefs(train_y):
    """
    This function calculates the mean coefficients and the bias for the model.

    This will drastically improve initial model performance.
    """
    mean = torch.ones(2)
    mean[0] = 1.0  # torch.quantile(train_y, torch.tensor([0.02, 0.98])).diff() / 2
    mean[1] = 0.1

    return mean


class SVGP(ApproximateGP):
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

        self.mean_module = gpytorch.means.LinearMean(2)
        self.mean_module.weights.data = mean_weights
        # self.mean_module.bias.data = mean_bias

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5, active_dims=(5, 6), ard_num_dims=2)
            * (
                gpytorch.kernels.MaternKernel(nu=0.5, active_dims=(3, 4))
                + gpytorch.kernels.ConstantKernel()
            )
        ) + gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=0.5, active_dims=(5, 6), ard_num_dims=2)
            * gpytorch.kernels.MaternKernel(nu=1.5, active_dims=2)
        )

    def forward(self, x):
        # print(x.shape)
        if x.dim() == 2:
            mean_x = self.mean_module(x[:, 0:2])
        else:
            mean_x = self.mean_module(x[:, :, 0:2])
        # print("OK")
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def load_data(data_directory, year="*", month="*", train_size=0.8, random_seed=42):
    """
    This function does a lot...it:
    1. Loads the data using the provided data directory.
    2. The load is split into training and test sets based on the list of stations.
    3. Features are extracted from the data, such as the periodized hour of the day.
    3. The data is then normalized to be fed into the model.
    """

    random.seed(random_seed)
    all_stations = glob.glob(os.path.join(data_directory, "station=*"))
    random.shuffle(all_stations)

    split_index = int(len(all_stations) * train_size)
    train_stations = all_stations[:split_index]
    test_stations = all_stations[split_index:]

    train_paths = get_all_paths(train_stations, year=year, month=month)
    test_paths = get_all_paths(test_stations, year=year, month=month)

    train_df = load_dataset(train_paths)
    test_df = load_dataset(test_paths)

    min_date = train_df["date"].min()
    train_df["hour"] = train_df["date"] - min_date
    test_df["hour"] = test_df["date"] - min_date

    train_df["hour"] = train_df["hour"].dt.total_seconds() / 3600
    test_df["hour"] = test_df["hour"].dt.total_seconds() / 3600

    periodize(train_df)
    periodize(test_df)

    train_X = train_df[
        ["t2m", "PC1", "hour", "sin_hour", "cos_hour", "lat", "lon"]
    ].values
    train_y = train_df["tempAvg"].values

    test_X = test_df[
        ["t2m", "PC1", "hour", "sin_hour", "cos_hour", "lat", "lon"]
    ].values
    test_y = test_df["tempAvg"].values

    assert train_X.shape[-1] == test_X.shape[-1]

    train_X = torch.tensor(train_X, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    test_X = torch.tensor(test_X, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32)

    return train_X, train_y, test_X, test_y, test_df


def periodize(df, period=24):
    df["sin_hour"] = np.sin(2 * df["hour"] * np.pi / period)
    df["cos_hour"] = np.cos(2 * df["hour"] * np.pi / period)


def load_dataset(paths):
    dfs = [pd.read_parquet(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)


def get_all_paths(station_path_list, month="*", year="*"):
    """
    This function finds all of the paths to the data files for
    a given set of stations, month, and year.
    """

    all_paths = []
    for station_path in station_path_list:
        all_paths.extend(
            glob.glob(
                os.path.join(station_path, f"year={year}/month={month}/*.parquet")
            )
        )
    return all_paths


def qce_coverage(y_samples, y_true, alpha=0.9):
    """
    Compute empirical coverage of central prediction intervals using PyTorch.

    Parameters
    ----------
    y_samples : torch.Tensor
        Shape (S, N) — predictive samples from the posterior
    y_true : torch.Tensor
        Shape (N,) — true labels
    alpha : float
        Desired coverage level (e.g., 0.9 for 90%)

    Returns
    -------
    coverage : float
        Fraction of test points with y_true inside predictive interval
    """
    # Compute quantiles across samples (dim=0 → across S samples per point)
    lower = torch.quantile(y_samples, q=(1 - alpha) / 2, dim=0)
    upper = torch.quantile(y_samples, q=(1 + alpha) / 2, dim=0)

    # Check if true values are inside the intervals
    inside = (y_true >= lower) & (y_true <= upper)

    return inside.float().mean().item()


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

    # Add a loss function with choices "ELBO" and "PLL"
    parser.add_argument(
        "--loss_function",
        type=str,
        default="ELBO",
        choices=["ELBO", "PLL"],
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

    args = parser.parse_args()
    main(args)
