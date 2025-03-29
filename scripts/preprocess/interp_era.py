"""
This script goes through all of the stations and agreggates all of the data into a CSV
file, so that the data can be analyzed.

"""

import os
import argparse

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

    # Create station_paths
    station_paths = [
        [args.input, args.output, s["stationId"], s["lat"], s["lon"]]
        for _, s in stations.iterrows()
    ]

    for i, row in enumerate(station_paths):
        # Print the progress
        print(f"Processing station {i + 1}/{len(station_paths)}")
        print(f"Station ID: {row[2]}")
        interpolate_era5_data(row)


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
        print("Processing month:", month)
        # Read the file
        ds = xr.open_dataset(os.path.join(input_dir, month), engine="cfgrib").load()

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
