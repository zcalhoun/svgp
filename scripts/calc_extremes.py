"""
This script calculates the number of hours above 35°C for each point in the test dataset.


"""

import os
import argparse

import numpy as np
import pandas as pd
import gpytorch
import torch
from torch.utils.data import DataLoader, TensorDataset

from Datasets import load_dataset
from Models import load_model, load_likelihood
from src.utils import init_inducing_points


def main(args):

    train_X, _, test_X, _, test_df, _, _ = load_dataset(
        "/work-old/zdc6/weather_underground/durham/combined/",
        variable="tempAvg",
        train_size=1.0,
        year=args.year,
        month=args.month,
        ref_data="/work-old/zdc6/era5_tabular/",
        upper_alpha=0.95,
        lower_alpha=0.01,
        calc_weights=False,
        normalize=False,
    )

    model_state_dict = torch.load(
        os.path.join(args.input, f"model_{args.year}_{args.month}.pt")
    )
    likelihood_state_dict = torch.load(
        os.path.join(args.input, f"likelihood_{args.year}_{args.month}.pt")
    )
    # model_state_dict = torch.load('/work-old/zdc6/temp_s_pll_2000/model_2024_1.pt')
    # likelihood_state_dict = torch.load('/work-old/zdc6/temp_s_pll_2000/likelihood_2024_1.pt')

    inducing_points = init_inducing_points(
        train_X,
        num_inducing_points=args.num_inducing_points,
    )

    model = load_model("tempAvg", inducing_points)
    likelihood = load_likelihood(args.likelihood)

    model.load_state_dict(model_state_dict)
    likelihood.load_state_dict(likelihood_state_dict)

    test_ds = TensorDataset(test_X)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model.eval()
    likelihood.eval()
    model = model.cuda()
    likelihood = likelihood.cuda()

    gt95 = []
    n_samples = args.num_samples
    with torch.no_grad():  # , gpytorch.settings.fast_pred_samples(state=True):
        for (X,) in test_loader:
            if torch.cuda.is_available():
                X = X.cuda()

            # We ditch the likelihood for now, as we are mainly interested in the
            # posterior variance from the model, and not really the noise model.
            preds = model(X)
            # mean.extend(preds.mean.cpu().numpy())
            # var.extend(preds.variance.cpu().numpy())

            # mean.extend(preds.mean.mean(dim=0).cpu().numpy())

            # sample = preds.sample()
            # upper.extend(torch.quantile(sample, 0.975, dim=0).cpu().numpy())
            # lower.extend(torch.quantile(sample, 0.025, dim=0).cpu().numpy())

            sample = preds.sample(
                torch.Size([n_samples])
            )  # sample_shape=torch.Size([n_samples]),)
            gt95.extend((sample > 35).int().detach().cpu().numpy().T)

    arr = ["lat", "lon"]
    arr.extend(range(0, n_samples))

    df = pd.DataFrame(
        np.column_stack([test_df["lat"].values, test_df["lon"].values, gt95]),
        columns=arr,
    )

    df.to_csv(
        os.path.join(args.input, f"gt95_{args.year}_{args.month}.csv"),
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the training and validation process."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the model directory."
    )
    parser.add_argument("--year", type=int, required=True, help="Year for the dataset.")
    parser.add_argument(
        "--month", type=int, required=True, help="Month for the dataset."
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to draw from the posterior.",
    )
    parser.add_argument(
        "--num_inducing_points",
        type=int,
        default=1000,
        help="Number of inducing points for the model.",
    )
    parser.add_argument(
        "--likelihood", type=str, default="Student", help="Type of likelihood to use."
    )

    parser.add_argument(
        "--batch_size", type=int, default=9900, help="Batch size for training."
    )

    arguments = parser.parse_args()
    main(arguments)
