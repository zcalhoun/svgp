"""
This script opens up a specific ERA file, and interpolates the data for a given set
of stations.
"""

import os
import argparse

import pandas as pd
import xarray as xr


def main(args):

    task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    file = get_file(int(task_id))
    print(f"{task_id}, File: {file}", flush=True)
    stations = get_stations(args.station_list)

    if args.test:
        print("Running in test mode")
        stations = stations[:2]
        print(f"Number of stations: {len(stations)}")
        print(f"Stations: {stations}")

    fp = os.path.join(args.input, file)

    interp_coords = xr.Dataset(
        {
            "latitude": (["points"], stations["lat"].values),
            "longitude": (["points"], stations["lon"].values),
        }
    )

    with xr.open_dataset(fp, engine="cfgrib") as ds:

        ds_interp = ds.interp(interp_coords, method="linear")

        df = (
            ds_interp.to_dataframe()
            .reset_index()
            .dropna()[
                ["valid_time", "latitude", "longitude", "t2m", "d2m", "u10", "v10"]
            ]
        )

    df = df.merge(
        stations[["lat", "lon", "stationId"]],
        left_on=["latitude", "longitude"],
        right_on=["lat", "lon"],
    )

    df = df.drop(columns=["latitude", "longitude"])

    # Save to CSV
    output_file = os.path.join(args.output, f"{file[:-5]}.csv")

    df.to_csv(output_file, index=False)


def get_stations(directory):
    """This function retrieves the list of stations from the manifest"""
    # manifest_path = os.path.join(directory, "durham_stations.csv")
    df = pd.read_csv(directory)
    # Remove stations that have not been QC'd
    # df = df[df["qcStatus"] == 1]
    return df


def get_file(task_id):
    """
    Get the file name from the task id.
    """
    date_range = pd.date_range(start="2019-01-01", end="2025-01-01", freq="M")
    formatted_dates = date_range.strftime("%Y-%m").tolist()
    return f"{formatted_dates[task_id]}.grib"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Interpolate the ERA5 data for the stations given."
    )

    # Add the following arguments:
    # -i, --input : the input directory
    # -o, --output : the output file
    parser.add_argument("-i", "--input", help="The ERA5 path", required=True)

    parser.add_argument("-o", "--output", help="The output directory", required=True)

    parser.add_argument(
        "-s",
        "--station_list",
        help="The list of stations to use",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--test",
        help="Run the script in test mode",
        action="store_true",
        default=False,
    )

    # Parse the arguments
    arguments = parser.parse_args()

    main(arguments)
