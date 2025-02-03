"""
This file contains code for loading the spatiotemporal data into
a torch tensor format, for use in the GPyTorch library.
"""

import os

import numpy as np
import pandas as pd
import rasterio as rio
import torch


def load_temperature_data(file_path, start_date, end_date, N=None):
    """
    This function loads just the temperature data.
    """
    # Create the list of dates to load
    dates = pd.date_range(start_date, end_date, freq="h")
    dates = dates.strftime("%Y-%m-%d %H:%M")

    # If there is a cutoff -- go ahead and only get the first N dates
    if N is not None:
        dates = dates[:N]

    temp_data = []
    spacetime_indices = []

    for i, d in enumerate(dates):
        date_path = os.path.join(file_path, "temp", f"temp_{d}.tif")

        if not os.path.exists(date_path):
            continue

        with rio.open(date_path) as src:
            data = src.read(1)

        shape = data.shape
        y = data[~np.isnan(data)]
        coords = np.indices(shape).reshape(2, -1).T
        coords = coords[~np.isnan(data.flatten())]

        coords = np.column_stack([np.ones(len(coords)) * i, coords])
        spacetime_indices.extend(coords)
        temp_data.extend(y)

    spacetime_indices = np.array(spacetime_indices)
    temp_data = np.array(temp_data)

    return torch.from_numpy(spacetime_indices), torch.from_numpy(temp_data)
