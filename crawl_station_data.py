"""
This script is useful for traversing and cataloging the data that we already have
collected in Weather Underground.
"""

import os
import sys
import json
import argparse
import pandas as pd


def main(arguments):
    """
    Run the main logic of collecting the data.
    """

    headers = [
        "stationId",
        "min_date_checked",
        "max_date_checked",
        "min_date_reported",
        "max_date_reported",
        "start_lat",
        "start_lon",
        "end_lat",
        "end_lon",
    ]
    fp = arguments.station_data
    stations = os.listdir(fp)

    # total = len(stations)
    all_data = []
    for station in simple_progress_bar(stations, prefix="Progress", size=50):

        station_data = {key: "" for key in headers}
        station_data["stationId"] = station

        station_path = os.path.join(fp, station)
        dates = os.listdir(station_path)
        if len(dates) == 0:
            all_data.append(station_data)
            continue
        dates = sorted([d for d in dates if len(d) == 13])
        station_data["min_date_checked"] = min(dates)
        station_data["max_date_checked"] = max(dates)

        for d in dates:
            with open(os.path.join(station_path, d)) as data:
                file = json.load(data)

            if file is None:
                continue
            else:
                station_data["start_lat"] = file["observations"][0]["lat"]
                station_data["start_lon"] = file["observations"][0]["lon"]
                station_data["min_date_reported"] = d

        for d in reversed(dates):
            with open(os.path.join(station_path, d)) as data:
                file = json.load(data)

            if file is None:
                continue
            else:
                station_data["end_lat"] = file["observations"][0]["lat"]
                station_data["end_lon"] = file["observations"][0]["lon"]
                station_data["max_date_reported"] = d

        all_data.append(station_data)
        if arguments.test:
            if len(all_data) > 10:
                break

    manifest = pd.DataFrame(all_data)
    manifest.to_csv(arguments.output, index=False)


def simple_progress_bar(iterable, prefix="", size=50):
    """
    A simple progress bar for the terminal.
    """
    total = len(iterable)
    for i, item in enumerate(iterable):
        x = int(size * (i + 1) / total)
        sys.stdout.write(f"\r{prefix}[{'#' * x}{'.' * (size - x)}] {i + 1}/{total}")
        sys.stdout.flush()
        yield item
    print()  # move to the next line


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Add file path to station data
    parser.add_argument(
        "--station_data",
        type=str,
        help="Path to the directory containing the station data",
        default="/work/zdc6/weather_underground/weather_underground/station_data",
    )

    # Add file path to where we would like to save the data
    parser.add_argument(
        "--output",
        type=str,
        help="Path to the directory where we would like to save the data",
        default="/work/zdc6/weather_underground/manifest.csv",
    )

    parser.add_argument(
        "--test", action="store_true", help="Run the script in test mode."
    )

    args = parser.parse_args()

    main(args)
