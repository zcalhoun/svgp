"""
This script contains the main training logic for each of the models.

Given a model name and a variable of interest, this script takes care of
loading the data, training the model, and saving the results.


Author: Zach Calhoun
Date: June 2025
"""

import os
import json
import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset

from Datasets import load_dataset
from Models import load_model, load_likelihood
from Trainers import train_model, validate_model, generate_maps
from src.utils import SimpleLogger, init_inducing_points, set_up_loss


def main(args):
    """
    Main function to run the training and validation process.
    """

    task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    logger = SimpleLogger(task_id)
    logger.info(args)
    year, month = parse_task_id(task_id)
    logger.info(f"Running task {task_id} for year {year} and month {month}")
    train_X, train_y, test_X, test_y, test_df = load_dataset(
        args.input,
        variable=args.variable,
        train_size=args.train_size,
        year=year,
        month=month,
        ref_data=args.ref_data,
        lower_alpha=args.lower_alpha,
        upper_alpha=args.upper_alpha,
    )

    logger.info(f"Loaded test_df with {len(test_df)} rows.")

    inducing_points = init_inducing_points(
        train_X,
        num_inducing_points=args.num_inducing_points,
    )

    model = load_model(args.variable, inducing_points)
    likelihood = load_likelihood(args.likelihood)

    mll = set_up_loss(args.loss, likelihood, model, train_y.size(0))

    train_ds = TensorDataset(train_X, train_y)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    logger.info("Starting training...")
    train_model(model, likelihood, mll, train_loader, args.num_epochs, args.lr)

    # Save the results to a JSON file.
    if os.path.exists(args.output) is False:
        os.makedirs(args.output)

    # Validate the model on the test set
    if args.train_size < 1.0:
        logger.info("Training completed. Now validating the model on the test set...")
        test_ds = TensorDataset(test_X, test_y)
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
        )
        results = validate_model(model, likelihood, test_loader)

        with open(
            os.path.join(args.output, f"results_{year}_{month}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(results, f)
    else:
        # If the training size is 1.0, we assume that we are saving the results.
        test_ds = TensorDataset(test_X)
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
        )

        results = generate_maps(model, likelihood, test_loader)

        test_df["pred"] = results["pred"]
        test_df["lower95"] = results["lower95"]
        test_df["upper95"] = results["upper95"]
        test_df["lower90"] = results["lower90"]
        test_df["upper90"] = results["upper90"]

        output_file = os.path.join(args.output, f"{year}-{month}.csv")
        test_df.to_csv(output_file, index=False)

    torch.save(
        model.state_dict(), os.path.join(args.output, f"model_{year}_{month}.pt")
    )
    torch.save(
        likelihood.state_dict(),
        os.path.join(args.output, f"likelihood_{year}_{month}.pt"),
    )


def parse_task_id(task_id):
    """
    Parse the task ID to extract the year and month.

    The task ID is expected to be the integer referring to the year/month
    since 2019-01, e.g., "0: 2019-01", "1: 2019-02", etc.

    """
    if task_id is None:
        raise ValueError("SLURM_ARRAY_TASK_ID environment variable is not set.")

    # Get list of years and months
    years = list(range(2019, 2025))
    months = list(range(1, 13))
    task_id = int(task_id)

    if task_id < 0 or task_id >= len(years) * len(months):
        raise ValueError(f"Invalid SLURM_ARRAY_TASK_ID: {task_id}")

    year = years[task_id // len(months)]
    month = months[task_id % len(months)]
    return year, month


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train a model for a given variable of interest."
    )
    parser.add_argument(
        "-i", "--input", help="The input directory containing the data", required=True
    )
    parser.add_argument(
        "-o", "--output", help="The output directory to save the results", required=True
    )
    parser.add_argument(
        "--variable",
        help="The variable of interest to train the model on",
        required=True,
        type=str,
        choices=["tempAvg", "dewptAvg"],
    )
    parser.add_argument(
        "--likelihood",
        help="The likelihood to use for the model",
        required=True,
        type=str,
        choices=["Gaussian", "Student"],
        default="Student",
    )

    parser.add_argument(
        "--loss",
        help="The loss function to use for the model",
        required=True,
        type=str,
        choices=["ELBO", "PLL"],
        default="PLL",
    )

    parser.add_argument(
        "--train_size",
        type=float,
        default=0.8,
        help="The proportion of the data to use for training (default: 0.8)",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="The number of epochs to train the model (default: 10)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="The batch size to use for training (default: 32)",
    )

    parser.add_argument(
        "--num_inducing_points",
        type=int,
        default=1000,
        help="The number of inducing points to use for the model (default: 1000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="The learning rate to use for the model (default: 0.01)",
    )

    parser.add_argument(
        "--ref_data",
        help="The path to the reference data file (default: None)",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--lower_alpha",
        help="The lower alpha value for filtering",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--upper_alpha",
        help="The upper alpha value for filtering",
        type=float,
        default=0.95,
    )

    arguments = parser.parse_args()

    main(arguments)
