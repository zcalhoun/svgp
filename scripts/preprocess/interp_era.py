"""
This script goes through all of the stations and agreggates all of the data into a CSV
file, so that the data can be analyzed.

"""

import os
import json
import argparse
import multiprocessing as mp

import pandas as pd
import xarray as xr


def main(args):

    # Obtain the list of stations in the input directory
    stations = get_stations(args.station_list)

    if args.test:
        print("Running in test mode")
        stations = stations[:10]
        print(f"Number of stations: {len(stations)}")
        print(f"Stations: {stations}")

    # Create the CSV file with the columns requested
    if args.cpu_count is None:
        cpu_count = mp.cpu_count()
    else:
        cpu_count = args.cpu_count

    print(f"Using {cpu_count} CPUs")

    # Create station_paths
    station_paths = [
        [args.input, args.output, s["stationId"], s["lat"], s["lon"]]
        for _, s in stations.iterrows()
    ]
    # Use multiprocessing to fetch the data
    with mp.Pool(processes=cpu_count) as pool:
        for result in pool.imap(interpolate_era5_data, station_paths):
            print(f"File created: {result}")


def get_stations(directory):
    """This function retrieves the list of stations from the manifest"""
    # manifest_path = os.path.join(directory, "durham_stations.csv")
    df = pd.read_csv(directory)
    # Remove stations that have not been QC'd
    # df = df[df["qcStatus"] == 1]
    return df


def interpolate_era5_data(args):
    """
    This function iterates over the ERA5 files and interpolates the data for the given station.

    The results are then saved to a CSV.
    """
    input_dir, output_dir, station_id, lat, lon = args

    era5_files = os.listdir(input_dir)

    station_results = []
    for month in era5_files:

        # Read the file
        ds = xr.open_dataset(os.path.join(input_dir, month), engine="cfgrib")

        # Select the data for the station
        ds = ds.interp(latitude=lat, longitude=lon, method="linear")

        # Convert to pandas dataframe
        df = (
            ds.to_dataframe()
            .reset_index()
            .dropna()[["valid_time", "t2m", "d2m", "u10", "v10"]]
        )

        # Save to CSV
        station_results.append(df)

    # Concatenate the results
    station_results = pd.concat(station_results)
    station_results["stationId"] = station_id

    # Save to CSV
    output_file = os.path.join(output_dir, f"{station_id}.csv")
    station_results.to_csv(output_file, index=False)
    return output_file


if __name__ == "__main__":
    # Set up the argparse
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

    # add argument for cpu_count
    parser.add_argument(
        "-c", "--cpu_count", help="The number of CPUs to use", default=None, type=int
    )

    parser.add_argument(
        "--test",
        help="Run the script in test mode",
        action="store_true",
        default=False,
    )

    # Parse the arguments
    arguments = parser.parse_args()

    # Run the main function
    main(arguments)
