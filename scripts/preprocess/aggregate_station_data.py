"""
This script goes through all of the stations and agreggates all of the data into a CSV
file, so that the data can be analyzed.

"""

import os
import json
import argparse
import multiprocessing as mp

import pandas as pd

COLUMNS = {
    "main": ["obsTimeUtc", "qcStatus", "lat", "lon", "solarRadiationHigh"],
    "metric": [
        "tempAvg",
        "dewptAvg",
        "precipTotal",
        "precipRate",
        "windspeedAvg",
        "windgustAvg",
    ],
}


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

    # Create the CSV file with headers
    # Create headers with text before . removed
    headers = create_header(COLUMNS)
    with open(args.output, "w") as f:
        f.write(",".join(headers) + "\n")

    # Create station_paths
    station_paths = [[args.input, s, args.start_date, args.end_date] for s in stations]
    # Use multiprocessing to fetch the data
    with mp.Pool(processes=cpu_count) as pool:
        for result in pool.imap(obtain_station_data, station_paths):
            if result is not None:
                with open(args.output, "a") as f:
                    f.writelines(result)


def get_stations(directory):
    """This function retrieves the list of stations from the manifest"""
    # manifest_path = os.path.join(directory, "durham_stations.csv")
    df = pd.read_csv(directory)
    # Remove stations that have not been QC'd
    # df = df[df["qcStatus"] == 1]
    return df["stationId"].tolist()


def obtain_station_data(args):
    """
    Read through the files and obtain the data for each station.
    """
    path, station, start_date, end_date = args

    # Create the list of files in the station
    files = os.listdir(os.path.join(path, station))

    if len(files) == 0:
        return None

    # Create list of files to check for
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    dates = [str(date).replace("-", "")[:8] + ".json" for date in dates]

    # Create the list of data
    data = ""
    for file in dates:
        fp = os.path.join(path, station, file)

        # If file doesn't exist, skip
        if not os.path.exists(fp):
            continue

        if os.path.getsize(fp) == 0:
            continue

        with open(fp) as f:
            # Read all of the columns from the observations:
            data_file = json.load(f)

        try:
            assert "observations" in data_file
        except TypeError:
            print(f"File {file} does not contain observations for station {station}.")
            continue

        if "observations" in data_file and len(data_file["observations"]) > 0:
            try:
                for obs in data_file["observations"]:
                    row = [station]
                    for col in COLUMNS["main"]:
                        row.append(str(obs[col]))

                    for col in COLUMNS["metric"]:
                        row.append(str(obs["metric"][col]))

                    data += ",".join(row) + "\n"
            except:
                print(f"Date {file} is not iterable for station {station}.")

    return data


def create_header(cols):
    return ["station"] + cols["main"] + cols["metric"]


if __name__ == "__main__":
    # Set up the argparse
    parser = argparse.ArgumentParser(description="Create a dataset from the data files")

    # Add the following arguments:
    # -i, --input : the input directory
    # -o, --output : the output file
    parser.add_argument("-i", "--input", help="The input directory", required=True)

    parser.add_argument("-o", "--output", help="The output file", required=True)

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

    # add start and end date arguments
    parser.add_argument(
        "-sd",
        "--start_date",
        help="The start date for the data",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-ed", "--end_date", help="The end date for the data", default=None, type=str
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
