"""
This script is useful for traversing and cataloging the data that we already have
collected in Weather Underground.
"""

import os
import sys
import json
import argparse
import multiprocessing as mp

import pandas as pd


def main(arguments):
    """
    Run the main logic of collecting the data.
    """

    fp = arguments.station_data
    stations = os.listdir(fp)

    if arguments.test:
        stations = stations[:100]

    process_args = [(fp, station) for station in stations]

    cpu_count = mp.cpu_count()
    print(f"There are {cpu_count} cores")
    all_data = []
    with mp.Pool(processes=cpu_count) as pool:
        for result in pool.imap_unordered(crawl_station, process_args):
            all_data.append(result)

    manifest = pd.DataFrame(all_data)
    manifest.to_csv(arguments.output, index=False)


def crawl_station(fp, station):
    """
    For each station, we are just going to crawl the data and return the
    relevant information.
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

    station_data = {key: "" for key in headers}
    station_data["stationId"] = station

    station_path = os.path.join(fp, station)
    dates = os.listdir(station_path)
    if len(dates) == 0:
        return station_data

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

    return station_data


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
