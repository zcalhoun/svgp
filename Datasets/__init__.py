"""
This file takes care of loading the specific dataset.

"""

from .datasets import load_data


def load_dataset(
    input_path,
    variable=None,
    train_size=1.0,
    year=None,
    month=None,
    ref_data=None,
    lower_alpha=0.01,
    upper_alpha=0.95,
    calc_weights=True,
):
    """
    Load the dataset from the specified input path.

    Args:
        input_path (str): Path to the dataset.
        variable (str, optional): Variable of interest. Defaults to None.
        train_size (float, optional): Proportion of data to use for training. Defaults to 1.0.

    Returns:
        tuple: Training and testing datasets.
    """
    if variable is None:
        raise ValueError("Variable must be specified.")
    if month is None or year is None:
        raise ValueError("Year and month must be specified.")
    if not (0 < train_size <= 1):
        raise ValueError("Train size must be between 0 and 1.")

    return load_data(
        root_dir=input_path,
        train_size=train_size,
        variable=variable,
        month=month,
        year=year,
        ref_data=ref_data,
        lower_alpha=lower_alpha,
        upper_alpha=upper_alpha,
        calc_weights=calc_weights,
    )
