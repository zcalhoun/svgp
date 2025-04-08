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
import faiss
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational.nearest_neighbor_variational_strategy import (
    NNVariationalStrategy,
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

    mean_weights, mean_bias = init_mean_coefs(train_y)

    if torch.cuda.is_available():
        logger.info("Using GPU for training")
        train_X = train_X.cuda()
        train_y = train_y.cuda()

    # Initialize the model

    logger.start_timer("INIT")

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = VNNGP(
        inducing_points=train_X,
        likelihood=likelihood,
        k=args.n_neighbors,
        training_batch_size=args.batch_size,
        mean_weights=mean_weights,
        mean_bias=mean_bias,
    )

    logger.stop_timer("INIT")

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    # Train the model

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mll = set_up_loss(args.loss_function, likelihood, model)

    test_ds = TensorDataset(test_X, test_y)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    epoch_losses = []
    best_mse = 1e6
    for epoch in range(args.num_epochs):

        logger.start_timer("TRAIN")
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        train_loss = train(model, likelihood, mll, optimizer, train_y)
        logger.stop_timer("TRAIN")

        logger.start_timer("VALIDATE")
        logger.info("Validating the model")
        val_prob, val_mse = validate(model, likelihood, test_loader)
        logger.stop_timer("VALIDATE")
        if epoch > 10:
            if val_mse < best_mse:
                best_mse = val_mse
                torch.save(model.state_dict(), args.output)
                torch.save(likelihood.state_dict(), args.output)

        logger.info(
            f"Epoch {epoch + 1}/{args.num_epochs} - "
            f"Train Loss: {train_loss:.3f} - "
            f"Validation Log Probability: {val_prob:.3f} - "
            f"Validation MSE: {val_mse:.3f}"
        )
        epoch_losses.append(epoch, train_loss, val_prob, val_mse)

    # After training, let's get predictions on all of the data
    preds = predict(model, likelihood, test_loader)
    test_df["preds"] = preds

    # Save the predictions to a CSV file
    test_df.to_csv(os.path.join(args.output, "predictions.csv"), index=False)

    epoch_losses = pd.DataFrame(
        epoch_losses, columns=["epoch", "train_loss", "val_prob", "val_mse"]
    )
    epoch_losses.to_csv(os.path.join(args.output, "epoch_losses.csv"), index=False)


def set_up_loss(loss_function, likelihood, model):
    """
    Set up the loss function for the model
    """

    if loss_function == "ELBO":
        mll = gpytorch.mlls.VariationalELBO(
            likelihood, model, num_data=model.num_inducing_points
        )
    elif loss_function == "PLL":
        mll = gpytorch.mlls.PredictiveLogLikelihood(
            likelihood, model, num_data=model.num_inducing_points
        )
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
    with torch.no_grad():

        for X, y in test_loader:
            if torch.cuda.is_available():
                X = X.cuda()

            pred = likelihood(model(X))
            preds.append(pred.mean.cpu().numpy())

    return np.concatenate(preds)


def validate(model, likelihood, test_loader):
    """
    Validate the model
    """

    model.eval()
    likelihood.eval()
    log_prob = 0
    mse = 0
    count = 0
    with torch.no_grad():

        for X, y in test_loader:
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()

            preds = likelihood(model(X))
            log_prob += preds.log_prob(y)
            mse += torch.sum((preds.mean - y) ** 2)
            count += y.size(0)
    log_prob /= count
    mse /= count

    return log_prob.item(), mse.item()


def train(model, likelihood, mll, optimizer, train_y):

    num_batches = model.variational_strategy._total_training_batches
    model.train()
    likelihood.train()

    epoch_loss = 0
    epoch_count = 0
    for i in range(num_batches):
        optimizer.zero_grad()
        output = model(x=None)
        current_training_indices = model.variational_strategy.current_training_indices
        y_batch = train_y[..., current_training_indices]

        if torch.cuda.is_available():
            y_batch = y_batch.cuda()

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
    mean[0] = torch.quantile(train_y, torch.tensor([0.02, 0.98])).diff() / 2
    mean[1] = 0.1

    return mean, train_y.mean()


class VNNGP(ApproximateGP):
    def __init__(
        self,
        inducing_points,
        likelihood,
        k=256,
        training_batch_size=256,
        mean_weights=None,
        mean_bias=None,
    ):
        """
        Define the VNNGP model
        """

        m, d = inducing_points.shape
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
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.LinearMean(2)
        self.mean_module.weights.data = mean_weights
        self.mean_module.bias.data = mean_bias

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5, active_dims=(5, 6), ard=2)
            * (
                gpytorch.kernels.MaternKernel(nu=0.5, active_dims=(3, 4))
                + gpytorch.kernels.ConstantKernel()
            )
        ) + gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=0.5, active_dims=(5, 6), ard=2)
            * gpytorch.kernels.MaternKernel(nu=1.5, active_dims=2)
        )

        self.likelihood = likelihood

    def forward(self, x):
        # print(x.shape)
        if x.dim() == 2:
            mean_x = self.mean_module(x[:, 0:2])
        else:
            mean_x = self.mean_module(x[:, :, 0:2])
        # print("OK")
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, prior=False, **kwargs):
        if x is not None:
            if x.dim() == 1:
                x = x.unsqueeze(-1)
        return self.variational_strategy(x=x, prior=False, **kwargs)


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

    x_max = train_X.max(dim=0)
    x_min = train_X.min(dim=0)

    train_X = 2 * (train_X - x_min.values) / (x_max.values - x_min.values) - 1
    test_X = 2 * (test_X - x_min.values) / (x_max.values - x_min.values) - 1

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
        "--n_neighbors", type=int, default=5, help="Number of neighbors"
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

    # Add a loss function with choices "ELBO" and "PLL"
    parser.add_argument(
        "--loss_function",
        type=str,
        default="ELBO",
        choices=["ELBO", "PLL"],
        help="Loss function to use",
    )

    args = parser.parse_args()
    main(args)
