"""
This script is responsible for breaking the data up into manageable chunks for
use in the training pipeline.
"""

import os
import argparse

import pandas as pd


def main(args):
    """
    Open up all of the files and merge them.

    """

    print("Loading the WU data", flush=True)
    wu = pd.read_csv(args.wu_file)
    wu = preprocess_wu_data(wu)
    print("WU data loaded, loading ERA next.", flush=True)
    era = open_era_files(args.era_dir)

    print("Now, loading the NLCD data.", flush=True)
    nlcd = pd.read_csv(args.nlcd_file)

    print("Merging the data.", flush=True)
    df = wu.merge(era, left_on=["station", "date"], right_on=["stationId", "obs_time"])

    # Add column for time in EST
    df["local_date"] = df["date"].dt.tz_convert("America/New_York")

    df["year"] = df["local_date"].dt.year
    df["month"] = df["local_date"].dt.month

    print("Lastly, merging the NLCD data.", flush=True)
    df = df.merge(nlcd, left_on=["station", "year"], right_on=["stationId", "year"])
    df = df[
        [
            "station",
            "year",
            "month",
            "lat",
            "lon",
            "date",
            "t2m",
            "d2m",
            "u10",
            "v10",
            "PC1",
            "PC2",
            "tempAvg",
            "dewptAvg",
            "solarRadiationHigh",
            "precipTotal",
            "precipRate",
            "windspeedAvg",
        ]
    ]

    print("Converting the temperature to Celsius.", flush=True)
    df["t2m"] = df["t2m"] - 273.15
    df["d2m"] = df["d2m"] - 273.15

    print("Data merged, now saving to parquet.", flush=True)
    df.to_parquet(
        args.destination_dir,
        engine="pyarrow",
        partition_cols=["station", "year", "month"],
        compression="snappy",
    )

    print("Data saved to preprocessed directory.", flush=True)


def preprocess_wu_data(wu):
    """
    This function preprocesses the weather underground data.
    """
    wu = wu[wu["qcStatus"] == 1]
    wu = wu[~wu["tempAvg"].isna()]
    # wu = wu[~wu["dewptAvg"].isna()]
    wu["date"] = pd.to_datetime(wu["obsTimeUtc"], utc=True).dt.round("H")

    return wu


def open_era_files(directory):
    """
    Iterate through the files, and read them into a single dataframe.
    """

    files = os.listdir(directory)

    dfs = []
    for file in files:
        path = os.path.join(directory, file)
        df = pd.read_csv(path)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    df["obs_time"] = pd.to_datetime(df["valid_time"], utc=True)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine training data from multiple files."
    )
    parser.add_argument(
        "--wu_file",
        type=str,
        required=True,
        help="File containing the weather underground data.",
    )
    parser.add_argument(
        "--era_dir",
        type=str,
        required=True,
        help="File containing the ERA5 data.",
    )
    parser.add_argument(
        "--nlcd_file",
        type=str,
        required=True,
        help="File containing the NLCD data.",
    )
    parser.add_argument(
        "--destination_dir",
        type=str,
        required=True,
        help="Path to the output combined parquet file.",
    )
    arguments = parser.parse_args()

    main(arguments)
