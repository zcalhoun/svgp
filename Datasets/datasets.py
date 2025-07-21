"""
This script contains the main logic for loading the datasets.
"""

import os
import re
import glob
import random

import numpy as np
import pandas as pd
from statsmodels.robust.scale import qn_scale
from scipy.stats import t
from sklearn.neighbors import KernelDensity

import torch
from .lgcp import lgcp_weight

REF_VARS = {"tempAvg": "t2m", "dewptAvg": "d2m"}


def load_data(
    root_dir,
    train_size=0.8,
    random_seed=42,
    variable=None,
    month=None,
    year=None,
    ref_data=None,
    upper_alpha=0.95,
    lower_alpha=0.01,
    calc_weights=True,
):
    """
    This is the generic code for loading the data for the training/validation
    pipeline.
    """

    random.seed(random_seed)
    all_stations = glob.glob(os.path.join(root_dir, "station=*"))
    random.shuffle(all_stations)

    if train_size != 1.0:
        split_index = int(len(all_stations) * train_size)
        train_stations = all_stations[:split_index]
        test_stations = all_stations[split_index:]

        train_paths = get_all_paths(train_stations, month=month, year=year)
        test_paths = get_all_paths(test_stations, month=month, year=year)

        train_df = load_dataframes(train_paths)
        test_df = load_dataframes(test_paths)

        train_df, test_df = run_qc(
            train_df,
            test_df=test_df,
            upper_alpha=upper_alpha,
            lower_alpha=lower_alpha,
            ref_var=REF_VARS[variable],
            wu_var=variable,
        )

    else:
        train_paths = get_all_paths(all_stations, month=month, year=year)
        train_df = load_dataframes(train_paths)

        train_df = run_qc(
            train_df,
            upper_alpha=upper_alpha,
            lower_alpha=lower_alpha,
            ref_var=REF_VARS[variable],
            wu_var=variable,
        )

        if month < 10:
            month = f"0{month}"
        test_df = pd.read_csv(os.path.join(ref_data, f"{year}-{month}.csv"))
        test_df["date"] = pd.to_datetime(test_df["obs_time"], utc=True)

    set_up_hours(train_df, test_df)

    train_X = train_df[
        [REF_VARS[variable], "PC1", "lat", "lon", "sin_hour", "cos_hour", "hour"]
    ].values
    train_y = train_df[variable].values
    test_X = test_df[
        [REF_VARS[variable], "PC1", "lat", "lon", "sin_hour", "cos_hour", "hour"]
    ].values
    if train_size != 1.0:
        test_y = test_df[variable].values

    # Only normalize the PC1 variable.
    mean = train_X[:, 1].mean(axis=0)
    std = train_X[:, 1].std(axis=0)
    train_X[:, 1] = (train_X[:, 1] - mean) / std
    test_X[:, 1] = (test_X[:, 1] - mean) / std

    # Convert to tensors
    train_X = torch.tensor(train_X, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    test_X = torch.tensor(test_X, dtype=torch.float32)
    if train_size != 1.0:
        test_y = torch.tensor(test_y, dtype=torch.float32)
    else:
        test_y = None

    # Create weights for the training set.
    if calc_weights:
        train_coords = np.unique(train_df[["lat", "lon"]].values, axis=0)
        test_coords = np.unique(test_df[["lat", "lon"]].values, axis=0)
        train_coords = torch.from_numpy(train_coords).float()
        test_coords = torch.from_numpy(test_coords).float()
        train_w, test_w = lgcp_weight(train_coords, test_coords)

        train_w = match_weights(train_coords, train_w, train_X[:, [2, 3]])
        test_w = match_weights(test_coords, test_w, test_X[:, [2, 3]])
    else:
        train_w = torch.ones(train_X.shape[0], dtype=torch.float32)
        test_w = torch.ones(test_X.shape[0], dtype=torch.float32)
    # weights = weight_features(train_df)

    # Normalize all of the other variables
    # mean = train_X[:, 1:].mean(dim=0)
    # std = train_X[:, 1:].std(dim=0)

    # train_X[:, 1:] = (train_X[:, 1:] - mean) / std
    # test_X[:, 1:] = (test_X[:, 1:] - mean) / std

    return train_X, train_y, test_X, test_y, test_df, train_w, test_w


def match_weights(coords, weights, features):
    """
    Match the weights to the features.
    """
    coord_dict = {
        tuple(coord.tolist()): val.item() for coord, val in zip(coords, weights)
    }

    # Match values
    matched_values = torch.tensor([coord_dict[tuple(q.tolist())] for q in features])

    # Convert to float32
    matched_values = matched_values.float()
    return matched_values


def weight_features(train_df):
    """
    This function creates weights for the features based on the location of the
    station.
    """

    features = np.unique(train_df[["lon", "lat"]].values, axis=0)
    kde = KernelDensity(kernel="exponential", bandwidth=0.01).fit(features)
    w = kde.score_samples(train_df[["lon", "lat"]].values)

    # Normalize the weights so they sum to the number of samples.
    w = 1 / np.exp(w)
    w = w / np.sum(w) * len(w)
    weights = torch.from_numpy(w).float()

    return weights


def run_qc(
    train_df,
    test_df=None,
    upper_alpha=0.95,
    lower_alpha=0.01,
    ref_var="t2m",
    wu_var="tempAvg",
):
    """
    This function runs through a few quality control steps on the dataframes.

    1. We remove rows with duplicate lat/lon/date combinations.
    2. We then run a statistical filter on the temperature data.
    3. Lastly, we remove rows where the filter removed more than 20% of the values.

    """

    # Step 1: Remove duplicate lat/lon/date combinations
    train_df = train_df[~train_df.duplicated(subset=["lat", "lon", "date"], keep=False)]
    if test_df is not None:
        test_df = test_df[
            ~test_df.duplicated(subset=["lat", "lon", "date"], keep=False)
        ]

    # Extra step for dewpoint...remove values that don't make sense.
    # Dew point should never be more than the temperature
    if wu_var == "dewptAvg":
        train_df = train_df[train_df["dewptAvg"] < train_df["tempAvg"]]
        train_df = train_df[~np.isnan(train_df["dewptAvg"])]
        if test_df is not None:
            test_df = test_df[test_df["dewptAvg"] < test_df["tempAvg"]]
            test_df = test_df[~np.isnan(test_df["dewptAvg"])]

    # Steps 2 & 3: Statistical filter on the temperature data.
    train_df["tempDiff"] = train_df[wu_var] - train_df[ref_var]
    if test_df is not None:
        test_df["tempDiff"] = test_df[wu_var] - test_df[ref_var]

    # Need to remove data where we don't have that much data.
    num_stations = train_df.groupby("date", as_index=False)["stationId"].count()
    valid_dates = num_stations[num_stations["stationId"] > 20]["date"]
    train_df = train_df[train_df["date"].isin(valid_dates)]

    ref_temps = pd.pivot_table(
        train_df, index="date", values="tempDiff", aggfunc=("median", qn_scale, "count")
    ).reset_index()

    # Note: we calculate critical values based on the t-distribution,
    # but in practice, this is very close to the normal distribution,
    # hence we demarcate the critical value as 'z'
    df = ref_temps["count"].mean()
    upper_cv = t.ppf(upper_alpha, df=df)
    lower_cv = t.ppf(lower_alpha, df=df)

    train_df = train_df.merge(ref_temps, left_on="date", right_on="date")
    train_df["z"] = (train_df["tempDiff"] - train_df["median"]) / train_df["qn_scale"]

    train_df["mask"] = (train_df["z"] > upper_cv).values + (
        train_df["z"] < lower_cv
    ).values
    mean_mask = train_df.groupby("stationId")["mask"].mean()
    # We hardcode 0.2 as the threshold for removing stations.
    # This is explicitly step 3 (which we do ahead of step 2).
    invalid_stations = mean_mask[mean_mask > 0.2].index
    train_df = train_df[~train_df["stationId"].isin(invalid_stations)]
    train_df = train_df[train_df["z"] < upper_cv]
    train_df = train_df[train_df["z"] > lower_cv]

    if test_df is not None:
        test_df = test_df.merge(ref_temps, left_on="date", right_on="date")
        test_df["z"] = (test_df["tempDiff"] - test_df["median"]) / test_df["qn_scale"]
        test_df["mask"] = (test_df["z"] > upper_cv).values + (
            test_df["z"] < lower_cv
        ).values
        mean_mask = test_df.groupby("stationId")["mask"].mean()
        invalid_stations = mean_mask[mean_mask > 0.2].index
        test_df = test_df[~test_df["stationId"].isin(invalid_stations)]
        test_df = test_df[test_df["z"] < upper_cv]
        test_df = test_df[test_df["z"] > lower_cv]

        return train_df, test_df

    # If not test_df, just return the train_df
    return train_df


def set_up_hours(train_df, test_df):
    """
    This function sets up the hour column in the train and test dataframes.
    """
    min_date = train_df["date"].min()
    train_df["hour"] = train_df["date"] - min_date
    test_df["hour"] = test_df["date"] - min_date

    train_df["hour"] = train_df["hour"].dt.total_seconds() / 3600
    test_df["hour"] = test_df["hour"].dt.total_seconds() / 3600

    periodize(train_df)
    periodize(test_df)


def periodize(df, period=24):
    """
    Create the sin and cos features for the hour column.
    """
    df["sin_hour"] = np.sin(2 * df["hour"] * np.pi / period)
    df["cos_hour"] = np.cos(2 * df["hour"] * np.pi / period)


def load_dataframes(paths):
    """
    Given a list of paths, just load and concatenate the dataframes.
    """
    dfs = []
    for p in paths:
        stationId = re.search(r"station=([^/]+)", p).group(1)
        df = pd.read_parquet(p)
        df["stationId"] = stationId
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def get_all_paths(station_path_list, month="*", year="*"):
    """
    This function just gets all the paths for the given month and year.
    """
    all_paths = []
    for station_path in station_path_list:
        all_paths.extend(
            glob.glob(
                os.path.join(station_path, f"year={year}/month={month}/*.parquet")
            )
        )
    return all_paths
