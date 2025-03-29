"""
This script opens up a specific ERA file, and interpolates the data for a given set
of stations.
"""

import os
import glob
import argparse
import multiprocessing as mp

import pandas as pd
import xarray as xr


def main():

    task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    file = get_file(int(task_id))
    print(file)


def get_file(task_id):
    """
    Get the file name from the task id.
    """
    date_range = pd.date_range(start="2019-01-01", end="2019-12-31", freq="M")
    formatted_dates = date_range.strftime("%Y-%m").tolist()
    return f"{formatted_dates[task_id]}.grib"


if __name__ == "__main__":
    main()
