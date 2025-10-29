"""
Orchestrates SVGP training/validation for spatiotemporal weather experiments.

Each SLURM array task corresponds to a (year, month) pair. For the assigned
period the script:

1. Loads the Weather Underground and reference ERA data.
2. Builds the sparse variational GP specified via CLI arguments.
3. Trains the model, optionally evaluates on a held-out split, and saves
   artifacts ready for downstream analysis.

Author: Zach Calhoun
Last modified: October 2025
"""

import os
import gc
import json
import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset
from gpytorch.likelihoods import GaussianLikelihood, StudentTLikelihood

from Datasets import load_dataset
from src import (
    SimpleLogger,
    init_inducing_points,
    set_up_loss,
    TempModel,
    DewpointModel,
    train_model,
    validate_model,
    generate_maps,
)


def main(args):
    """
    Run the full training/evaluation workflow for a single (year, month) task.

    The workflow resolves the date from SLURM, prepares the tensors required
    by the SVGP model, trains with the chosen loss, and finally persists the
    fitted state plus optional evaluation outputs.
    """

    task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    logger = SimpleLogger(task_id)
    logger.info(args)
    year, month = parse_task_id(task_id)
    logger.info(f"Running task {task_id} for year {year} and month {month}")

    train_X, train_y, test_X, test_y, test_df, train_w, test_w = load_dataset(
        args.input,
        variable=args.variable,
        train_size=args.train_size,
        year=year,
        month=month,
        ref_data=args.ref_data,
        lower_alpha=args.lower_alpha,
        upper_alpha=args.upper_alpha,
        normalize=args.normalize,
    )

    logger.info(f"Loaded test_df with {len(test_df)} rows.")

    inducing_points = init_inducing_points(
        train_X,
        num_inducing_points=args.num_inducing_points,
    )

    model = load_model(args.variable, inducing_points)
    likelihood = load_likelihood(args.likelihood)

    mll = set_up_loss(args.loss, likelihood, model, train_y.size(0))

    train_ds = TensorDataset(train_X, train_y, train_w)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    likelihood = likelihood.to(device)

    logger.info("Starting training...")
    train_model(model, likelihood, mll, train_loader, args.num_epochs, args.lr)

    # Save the results to a JSON file.
    if os.path.exists(args.output) is False:
        os.makedirs(args.output)

    # Clean up some of the memory
    del train_X, train_y, train_w, train_ds, train_loader
    gc.collect()
    torch.cuda.empty_cache()

    torch.save(
        model.state_dict(), os.path.join(args.output, f"model_{year}_{month}.pt")
    )
    torch.save(
        likelihood.state_dict(),
        os.path.join(args.output, f"likelihood_{year}_{month}.pt"),
    )

    # Validate the model on the test set
    if args.train_size < 1.0:
        logger.info("Training completed. Now validating the model on the test set...")
        test_ds = TensorDataset(test_X, test_y, test_w)
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


def parse_task_id(task_id):
    """
    Map a SLURM array task identifier to its corresponding (year, month).

    The mapping enumerates months sequentially starting at 2019-01 with id=0.
    Raises:
        ValueError: if the identifier is missing or outside the supported range.
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


def load_model(variable, inducing_points):
    """
    Loads the model based on the variable and inducing points.
    """
    if variable == "tempAvg":
        return TempModel(inducing_points)
    elif variable == "dewptAvg":
        return DewpointModel(inducing_points)
    else:
        raise ValueError(f"Unsupported variable: {variable}. Please use 'tempAvg'.")


def load_likelihood(likelihood):
    """
    Set up the likelihood for the model
    """
    if likelihood == "Gaussian":
        return GaussianLikelihood()

    if likelihood == "Student":
        return StudentTLikelihood()

    raise ValueError(f"Unknown likelihood: {likelihood}")


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
        choices=["ELBO", "PLL", "W-PLL"],
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

    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Whether to normalize the data (default: False)",
    )

    arguments = parser.parse_args()

    main(arguments)
