"""
This script contains the code to fit the VNNGP model to the dataset.

Args:
    dataset: The dataset to fit the model to.
    sampling_method: The method to use to split the data.
    k: The number of neighbors to use in the nearest neighbor strategy.
    training_batch_size: The batch size to use when training the model.
    checkpoint_path: The path to save the model to.
    output_path: The path to save the log files to.
"""

import os
import argparse
import logging
import pandas as pd

import gpytorch
import torch
from torch.utils.data import TensorDataset, DataLoader

import Datasets
import models


def main(args):
    """
    The main function which loads the data, the model, and runs the train/validation
    loop. The arguments are passed in from the command line.
    """
    # Check if the output path exists, if not create it
    validate_output_path(args.output_path)

    set_up_logger(args.log_level, args.output_path)

    logging.info(args)

    train_X, train_y, val_X, val_y, period = load_data(args.dataset, args.file_path)

    if args.smoke_test:
        # Only use a small subset of the training/validation data
        train_X = train_X[:10000]
        train_y = train_y[:10000]
        val_X = val_X[:10000]
        val_y = val_y[:10000]
    # Check if train_X is contiguous
    if not train_X.is_contiguous():
        train_X = train_X.contiguous()

    # Check the checkpoints directory to see if there is a base
    # model with the given k and batch_size already saved.
    # If there is, load the model and continue training.

    if not os.path.exists(args.checkpoint_path):
        logging.info("Creating directory %s", args.checkpoint_path)
        os.makedirs(args.checkpoint_path)

    # Create the likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(args.noise_constraint)
    )

    logging.info("Creating model with k=%d", args.k)
    model = models.load(
        args.model,
        train_X,
        likelihood,
        k=args.k,
        training_batch_size=args.training_batch_size,
    )

    # If cuda is available, add to CUDA
    if torch.cuda.is_available():
        logging.info("CUDA is available")
        likelihood = likelihood.cuda()
        model = model.cuda()
    else:
        logging.info("CUDA is not available")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=args.gamma)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))
    best_mse = float("inf")
    epochs = args.epochs

    # Initialize arrays to store train/val losses
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        logging.info("Epoch %d", epoch)
        train_loss = train(model, likelihood, mll, optim, train_X, train_y)
        val_loss = validate(model, likelihood, val_X, val_y)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_mse:
            best_mse = val_loss
            torch.save(
                model.state_dict(), os.path.join(args.checkpoint_path, "model.pth")
            )
        scheduler.step()
        logging.info(
            "Epoch %d - Train Loss: %f - Val Loss: %f", epoch, train_loss, val_loss
        )

    # Create a dataframe from the train/val losses
    losses = pd.DataFrame({"train_loss": train_losses, "val_loss": val_losses})

    # Save the losses to a csv file
    losses.to_csv(os.path.join(args.output_path, "losses.csv"))


def validate_output_path(output_path):
    """
    This function checks if the output path exists, and if not creates it.
    """
    if not os.path.exists(output_path):
        logging.info("Creating directory %s", output_path)
        os.makedirs(output_path)


def load_data(dataset, file_path):
    """
    This function handles loading the dataset, and returns the
    training and validation data.
    """

    dataset = Datasets.load(dataset, file_path)

    # Split the dataset into a training and validation set
    train_X, train_y = dataset.get_train()
    val_X, val_y = dataset.get_val()

    # Convert data to torch tensors and add to cuda if available
    train_X = torch.from_numpy(train_X).float()
    train_y = torch.from_numpy(train_y).float()
    val_X = torch.from_numpy(val_X).float()
    val_y = torch.from_numpy(val_y).float()

    # Use min/max normalization for the X values.
    x_max = train_X.max(dim=0)
    x_min = train_X.min(dim=0)

    train_X = 1000 * (train_X - x_min.values) / (x_max.values - x_min.values) - 1
    val_X = 1000 * (val_X - x_min.values) / (x_max.values - x_min.values) - 1

    period = 1000 * dataset.period / (x_max.values[0] - x_min.values[0])

    if torch.cuda.is_available():
        train_X, train_y = train_X.cuda(), train_y.cuda()
        val_X, val_y = val_X.cuda(), val_y.cuda()
        train_X = train_X.contiguous()
        val_X = val_X.contiguous()

    return train_X, train_y, val_X, val_y, period


def validate(model, likelihood, val_X, val_y):
    """
    Run the validation loop and return the MSE loss.
    """
    val_dataset = TensorDataset(val_X.float(), val_y.float())
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    model.eval()
    likelihood.eval()
    means = torch.tensor([0.0])
    val_mse = 0
    count = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
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


def set_up_logger(log_level, output_path):
    """
    Set up the logger to log to a file.
    """
    logging.basicConfig(
        level=log_level,
        filemode="w",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename=os.path.join(output_path, "output.log"),
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="UNC",
        help="The dataset to fit the model to.",
    )

    parser.add_argument(
        "--file_path",
        type=str,
        help="The path to the dataset.",
    )

    parser.add_argument(
        "--sampling_method",
        type=str,
        default="chunk_by_sensor",
        help="The method to use to split the data.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="BaseVNNGP",
        help="The model to use.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=256,
        help="The number of neighbors to use in the nearest neighbor strategy.",
    )

    parser.add_argument(
        "--target",
        type=str,
        default="temperature",
        help="The target variable to predict.",
    )

    parser.add_argument(
        "--training_batch_size",
        type=int,
        default=256,
        help="The batch size to use when training the model.",
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints",
        help="The path to save the model to.",
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
        "--gamma",
        type=float,
        default=0.9,
        help="The gamma value to use when training the model.",
    )

    parser.add_argument(
        "--noise_constraint",
        type=float,
        default=1e-6,
        help="The noise constraint to use when training the model.",
    )

    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run a smoke test to make sure the code runs.",
    )

    args = parser.parse_args()

    main(args)
