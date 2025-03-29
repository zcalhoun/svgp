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

    # Get the slurm environmental variable for the task id
    task_id = os.getenv("SLURM_ARRAY_TASK_ID")

    print(task_id)


if __name__ == "__main__":
    main()
