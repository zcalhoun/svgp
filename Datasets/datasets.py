"""
This script contains the main logic for loading the datasets.
"""

import os
import glob
import random

import numpy as np
import pandas as pd
import torch


def load_data(
    root_dir,
    train_size=0.8,
    random_seed=42,
    variable=None,
    month=None,
    year=None,
):
    """
    This is the generic code for loading the data for the training/validation
    pipeline.
    """

    random.seed(random_seed)
    all_stations = glob.glob(os.path.join(root_dir, "station=*"))
    random.shuffle(all_stations)

    split_index = int(len(all_stations) * train_size)
    train_stations = all_stations[:split_index]
    test_stations = all_stations[split_index:]

    train_paths = get_all_paths(train_stations, month=month, year=year)
    test_paths = get_all_paths(test_stations, month=month, year=year)

    train_df = load_dataframes(train_paths)
    test_df = load_dataframes(test_paths)

    set_up_hours(train_df, test_df)

    if variable is "tempAvg":
        train_X = train_df[
            ["t2m", "PC1", "lat", "lon", "sin_hour", "cos_hour", "hour"]
        ].values
        train_y = train_df["tempAvg"].values
        test_X = test_df[
            ["t2m", "PC1", "lat", "lon", "sin_hour", "cos_hour", "hour"]
        ].values
        test_y = test_df["tempAvg"].values

        mean = train_X[:, 1].mean(axis=0)
        std = train_X[:, 1].std(axis=0)
        train_X[:, 1] = (train_X[:, 1] - mean) / std
        test_X[:, 1] = (test_X[:, 1] - mean) / std

    else:
        raise ValueError(f"Unsupported variable: {variable}. Please use 'tempAvg'.")
    # Convert to tensors
    train_X = torch.tensor(train_X, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    test_X = torch.tensor(test_X, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32)

    return train_X, train_y, test_X, test_y


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
        df = pd.read_parquet(p)
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
