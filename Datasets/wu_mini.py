"""
This dataloader takes care of the preprocessing for dealing with the Weather Underground
dataset.
"""

import pandas as pd
import numpy as np


class WeatherUnderground:
    """
    A basic class to load the Weather Underground data and prepare it for use in
    a training pipeline.
    """

    def __init__(
        self,
        file_path: str,
        train_hours: int = 480,
        percent_hold_out: float = 0.2,
        period: int = 24,
        random_seed: int = 42,
    ):
        """
        Arguments:
        file_path: str
            The path to the file containing the data.
        train_hours: int
            The number of hours to keep in the training set.
        percent_hold_out: float
            The percentage of sensors to hold out for the validation set.
        """
        self.period = period
        df = pd.read_csv(file_path)

        # First, we split based on hours
        train = df[df["hour"] < train_hours]
        np.random.seed(random_seed)
        self.train, self.val = self._hold_out_sensors(train, percent_hold_out)
        self.val_future = df[df["hour"] >= train_hours]

    def _hold_out_sensors(self, df, hold_out_sensors):
        df["key"] = [
            "-".join([str(x), str(y)]) for x, y in zip(df["x"].values, df["y"].values)
        ]

        # Let's keep out a subset of the sensors for the validation set.
        sensors = df["key"].unique()
        hold_out = np.random.choice(
            sensors, int(hold_out_sensors * len(sensors)), replace=False
        )

        val_df = df[df["key"].isin(hold_out)]
        train_df = df[~df["key"].isin(hold_out)]

        return train_df, val_df

    def get_train(self):
        """
        Set up the covariates to be minutes, latitude, and longitude.
        """

        X = self.train[["hour", "x", "y"]].values
        X = self._periodize(X)
        y = self.train["temp"].values

        return X, y

    def _periodize(self, X):
        sin_term = self._add_sin_term(X[:, 0])
        cos_term = self._add_cos_term(X[:, 0])
        X = np.column_stack((X, sin_term, cos_term))
        return X

    def _add_sin_term(self, X):
        return np.sin(2 * np.pi * X / self.period)

    def _add_cos_term(self, X):
        return np.cos(2 * np.pi * X / self.period)

    def get_val(self):
        """
        Return the validation data, after normalizing the y values based
        on the training data.
        """

        X = self.val[["hour", "x", "y"]].values
        X = self._periodize(X)
        y = self.val["temp"].values

        return X, y

    def get_val_future(self):
        """
        Return the validation data, after normalizing the y values based
        on the training data.
        """

        X = self.val_future[["hour", "x", "y"]].values
        X = self._periodize(X)
        y = self.val_future["temp"].values

        return X, y
