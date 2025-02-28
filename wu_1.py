"""
Experiment: WU-1
Author: Zach Calhoun
Date: 2025-02-25
Description:
    This script contains the logic needed to run the first experiment,
    in which we would like to understand the effect of the constant used as well as the 
    nearest neighbor count on the performance of the model. The hypothesis is that there
    is an ideal constant given the number of nearest neighbors that will yield the best
    results for the model.

    This script:
    1. Loads the experimental parameters.
    2. Loads the data.
    3. Initializes and fits the model.
    4. Saves the experimental results (but not the model itself).

    We want to have saved:
    1. Model parameters.
    2. Train, test, and val loss.
    3. The time it took to run the experiment.
    4. Generated figures for prediction.

    Experimental parameters:
    1. Constant C (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
"""

import os
import sys
import csv
import logging
import argparse

import torch
import gpytorch
from torch.utils.data import TensorDataset, DataLoader

import Datasets
import models


def main(args):
    """
    Run the main logic for the experiment, in which we just need to train the model
    on multiple constants.
    """
    validate_output_path(args.output_path)

    set_up_logger(args.log_level)

    if args.smoke_test:
        sys.stdout.reconfigure(line_buffering=True)

    logging.info(args)

    ######
    # Set up experiment
    ######
    if args.smoke_test:
        constants = [1, 2]
    else:
        constants = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    initialize_csv(args.output_path)

    for C in constants:
        logging.info("Running with C=%d", C)
        train_loss, val_loss, test_loss, noise = run_experiment(args, C)

        update_csv(
            args.output_path,
            [C, train_loss, val_loss, test_loss, noise],
        )


def run_experiment(args, C):
    """
    Run the experiment with the given constant.
    """

    #####
    # Load the data
    #####
    if args.smoke_test:
        wu_data = Datasets.load("WU_mini", args.file_path, train_hours=10)
    else:
        wu_data = Datasets.load("WU_mini", args.file_path)
    X_train, y_train = wu_data.get_train()
    X_val, y_val = wu_data.get_val()
    X_test, y_test = wu_data.get_val_future()

    # Normalize the data
    X_max = X_train.max(0)
    X_min = X_train.min(0)
    X_train = 2 * (X_train - X_min) / (X_max - X_min) - 1
    X_val = 2 * (X_val - X_min) / (X_max - X_min) - 1
    X_test = 2 * (X_test - X_min) / (X_max - X_min) - 1

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Multiple X_train by C
    X_train[:, 0] *= C
    X_val[:, 0] *= C
    X_test[:, 0] *= C

    if torch.cuda.is_available():
        X_train = X_train.cuda()
        # y_train = y_train.cuda()
    if not X_train.is_contiguous():
        X_train = X_train.contiguous()

    #####
    # Fit the model
    #####
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    inducing_point_prior = gpytorch.distributions.MultivariateNormal(
        y_train.clone(),
        torch.eye(y_train.size(0)) * args.inducing_point_prior_variance,
    )

    logging.info("Initializing model.")
    model = models.load(
        "BaseVNNGP_PeriodicFeatures",
        X_train,
        k=args.num_neighbors,
        training_batch_size=args.batch_size,
        inducing_point_prior=inducing_point_prior,
    )
    logging.info("Model initialized.")
    sys.stdout.flush()

    # Initialize the model
    # Good initialization greatly expedites the training process!
    # TODO: Consider how to initialize the model better.
    model.mean_module.constant = y_train.mean()
    model.covar_module.kernels[0].outputscale = 10
    model.covar_module.kernels[1].outputscale = 15
    model.covar_module.kernels[2].outputscale = 1
    model.covar_module.kernels[3].outputscale = 1
    likelihood.noise = 1

    # If cuda is available, add to CUDA
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    optim = torch.optim.Adam(
        [
            {
                "params": model.parameters(),
                "lr": args.lr,
            },
            {
                "params": likelihood.parameters(),
                "lr": args.lr,
            },
        ]
    )
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, y_train.numel())

    best_mse = float("inf")

    for epoch in range(args.epochs):
        logging.info("Starting epoch %d", epoch)

        train_loss = train(model, likelihood, mll, optim, X_train, y_train)
        val_loss = validate(model, likelihood, X_val, y_val, args.batch_size)

        logging.info(
            "Epoch %d - Train Loss:` %f - Val Loss: %f", epoch, train_loss, val_loss
        )
        # Flush the logs to the file.
        sys.stdout.flush()

        if val_loss < best_mse:
            best_mse = val_loss
        else:
            logging.info("Early stopping.")
            break

    test_loss = validate(model, likelihood, X_test, y_test, args.batch_size)

    return train_loss, val_loss, test_loss, likelihood.noise.item()


def initialize_csv(output_path):
    """
    This function initializes the csv file to write the model to.
    """
    headers = ["C", "train_loss", "val_loss", "test_loss", "noise"]
    with open(os.path.join(output_path, "results.csv"), "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)


def update_csv(output_path, row):
    """
    This function updates the csv file with the given row.
    """
    with open(os.path.join(output_path, "results.csv"), "a", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def validate_output_path(output_path):
    """
    This function checks if the output path exists, and if not creates it.
    """
    if not os.path.exists(output_path):
        logging.info("Creating directory %s", output_path)
        os.makedirs(output_path)


def set_up_logger(log_level):
    """
    Set up the logger to log to a file.
    """
    logging.basicConfig(
        level=log_level,
        filemode="w",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

    logging.captureWarnings(True)


def validate(model, likelihood, val_X, val_y, batch_size):
    """
    Run the validation loop and return the MSE loss.
    """
    val_dataset = TensorDataset(val_X.float(), val_y.float())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    likelihood.eval()
    means = torch.tensor([0.0])
    val_mse = 0
    count = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            if torch.cuda.is_available():
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
            preds = model(x_batch)
            means = torch.cat([means, preds.mean.cpu()])

            diff = torch.pow(preds.mean - y_batch, 2)
            val_mse += diff.sum()
            count += y_batch.size(0)

    return val_mse.item() / count


def train(model, likelihood, mll, optim, train_X, train_y):
    """
    Run the training loop over the data.

    Args:
        model: The model to train.
        likelihood: The likelihood to use.
        mll: The loss function to use.
        optim: The optimizer to use.
        train_X: The training data to use.
        train_y: The training labels to use.

    Note: there is no need for training data, as the the training points are stored
    as inducing points in the given model.

    Returns:
        The mean squared error loss on the training data.
    """
    model.train()
    likelihood.train()

    num_batches = (
        train_X.size(0) + model.variational_strategy.training_batch_size - 1
    ) // model.variational_strategy.training_batch_size
    logging.info("Training with %d batches", num_batches)

    mse_loss = 0
    count = 0
    for i in range(num_batches):
        optim.zero_grad()
        output = model(x=None)
        current_training_indices = model.variational_strategy.current_training_indices
        y_batch = train_y[..., current_training_indices]
        if torch.cuda.is_available():
            y_batch = y_batch.cuda()
        loss = -mll(output, y_batch)
        loss.backward()
        optim.step()
        if i % 100 == 0:
            logging.info("Iter %d/%d - Loss: %f", i, num_batches, loss.item())

        mse_loss += (output.mean - y_batch).pow(2).sum().item()
        count += y_batch.size(0)

    return mse_loss / count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file_path",
        type=str,
        help="The path to the dataset.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="The batch size to use when training the model.",
    )

    parser.add_argument(
        "-k",
        "--num_neighbors",
        type=int,
        default=64,
        help="The number of nearest neighbors to use.",
        metavar="k",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="output",
        help="The path to save the log files to.",
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="The logging level to use.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="The number of epochs to train the model for.",
    )

    # Add learning rate, gamma, and other hyperparameters
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="The learning rate to use when training the model.",
    )

    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run a smoke test to make sure the code runs.",
    )

    parser.add_argument(
        "--inducing_point_prior_variance",
        type=float,
        default=4.0,
        help="The variance to use for the inducing point prior.",
    )

    arguments = parser.parse_args()

    main(arguments)
