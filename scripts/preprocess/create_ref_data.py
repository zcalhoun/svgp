"""
This script deals with preformatting data for input into the model.

Basically, we take in the ERA5 data and the NLCD data, then organize the data
into a tabular format so that it can be fed into the model.

"""

import os
import argparse

import numpy as np
import pandas as pd
import xarray as xr


def main(args):

    task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    year, month = parse_task_id(task_id)

    if month < 10:
        month_str = f"0{month}"
    else:
        month_str = str(month)
    ds = xr.open_dataset(
        os.path.join(args.input, f"{year}-{month_str}.grib"), engine="cfgrib"
    )

    new_lat = np.arange(ds.latitude.max(), ds.latitude.min(), -0.005)
    new_lon = np.arange(ds.longitude.min(), ds.longitude.max(), 0.005)

    ds_interp = ds.interp(latitude=new_lat, longitude=new_lon, method="linear")

    ds_interp = (
        ds_interp.to_dataframe()
        .reset_index()
        .dropna()[["valid_time", "latitude", "longitude", "t2m", "d2m", "u10", "v10"]]
    )

    ds_interp["obs_time"] = pd.to_datetime(ds_interp["valid_time"], utc=True)

    ds_interp["year"] = ds_interp["obs_time"].dt.year

    pc_map = pd.read_csv(args.nlcd)

    merged = ds_interp.merge(
        pc_map,
        left_on=["year", "latitude", "longitude"],
        right_on=["year", "lat", "lon"],
    )

    merged = merged.drop(columns=["lat", "lon", "valid_time", "obs_time"])

    merged["t2m"] = merged["t2m"] - 273.15  # Convert from Kelvin to Celsius
    merged["d2m"] = merged["d2m"] - 273.15  # Convert from Kelvin to Celsius

    output_file = os.path.join(args.output, f"{year}-{month_str}.csv")
    merged.to_csv(output_file, index=False)


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
        description="Preprocess the ERA5 and NLCD data for model input."
    )

    parser.add_argument(
        "-i", "--input", help="The input directory containing ERA5 data", required=True
    )
    parser.add_argument(
        "-o",
        "--output",
        help="The output directory for preprocessed data",
        required=True,
    )
    parser.add_argument(
        "--nlcd",
        help="The path to the NLCD reference file",
        required=True,
        type=str,
    )

    args = parser.parse_args()
    main(args)
