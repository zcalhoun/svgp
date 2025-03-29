"""
This script goes through all of the stations and agreggates all of the data into a CSV
file, so that the data can be analyzed.

"""

import os
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

    # Create station_paths
    station_paths = [
        [args.input, args.output, s["stationId"], s["lat"], s["lon"]]
        for _, s in stations.iterrows()
    ]

    for i, row in enumerate(station_paths):
        # Print the progress
        print(f"Processing station {i + 1}/{len(station_paths)}")
        print(f"Arguments: {row}")
        # print the time
        print(f"Time: {pd.Timestamp.now()}")
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

    process_args = [[os.path.join(input_dir, fp), lat, lon] for fp in era5_files]
    print(process_args)
    # return None
    cpu_count = mp.cpu_count()

    # Use multiprocessing to fetch the data
    with mp.Pool(processes=cpu_count) as pool:
        for result in pool.imap(extract_data, process_args):
            station_results.append(result)

    # Concatenate the results
    station_results = pd.concat(station_results)
    station_results["stationId"] = station_id

    # Save to CSV
    output_file = os.path.join(output_dir, f"{station_id}.csv")
    station_results.to_csv(output_file, index=False)


def extract_data(args):
    """
    Open the file and extract the data for the given station.
    """

    era_fps, lat, lon = args

    for fp in era_fps:
        # Read the file
        with xr.open_dataset(fp, engine="cfgrib") as ds:

            # Select the data for the station
            ds_point = ds.interp(latitude=lat, longitude=lon, method="linear")

            # Convert to pandas dataframe
            df = (
                ds_point.to_dataframe()
                .reset_index()
                .dropna()[["valid_time", "t2m", "d2m", "u10", "v10"]]
            )

    return df


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
