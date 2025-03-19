"""
This script handles pulling the ERA5 data and saving it to a local directory.
"""

import os
import argparse
from calendar import monthrange
import cdsapi


def main(args):
    """
    Iterate through the years/months and make the request.
    """

    year = args.year

    months = [f"{month:02d}" for month in range(1, 13)]

    # for year in years:
    days = get_days_by_month(year)
    for month in months:
        print(f"Downloading data for {year}-{month}", flush=True)
        month_days = days[month]
        make_request(year, month, month_days)


def make_request(year, month, days):
    """
    Make the request to the CDS API.
    """

    dataset = "reanalysis-era5-land"
    request = {
        "variable": [
            "2m_dewpoint_temperature",
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        "year": str(year),
        "month": month,
        "day": days,
        "time": [
            "00:00",
            "01:00",
            "02:00",
            "03:00",
            "04:00",
            "05:00",
            "06:00",
            "07:00",
            "08:00",
            "09:00",
            "10:00",
            "11:00",
            "12:00",
            "13:00",
            "14:00",
            "15:00",
            "16:00",
            "17:00",
            "18:00",
            "19:00",
            "20:00",
            "21:00",
            "22:00",
            "23:00",
        ],
        "data_format": "grib",
        "download_format": "unarchived",
        "area": [36.23, -79.05, 35.75, -78.55],
    }

    client = cdsapi.Client()
    # print(request)

    target = os.path.join(DEST_DIR, f"{year}-{month}.grib")
    client.retrieve(dataset, request, target)  # .download()


def get_days_by_month(year):
    """
    Returns the days of the month
    """
    days_by_month = {}
    for month in range(1, 13):
        num_days = monthrange(year, month)[1]  # Get number of days for the given month
        days_list = [
            f"{day:02d}" for day in range(1, num_days + 1)
        ]  # Format days as "01", "02", etc.
        days_by_month[f"{month:02d}"] = days_list  # Format month as "01", "02", etc.
    return days_by_month


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--year",
        type=int,
        help="Year to pull data for. Default is 2019.",
        default=2019,
    )
    arguments = parser.parse_args()
    main(arguments)
