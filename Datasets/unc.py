"""
This module contains the code to load the UNC dataset.
"""

import pandas as pd
import numpy as np


pd.options.mode.chained_assignment = None


class UNC_Dataset:
    """This class handles loading the UNC data and dividing into a training,
    validation, and test set."""

    def __init__(
        self,
        file_path,
        sampling_method="chunk_by_sensor",
        replicates=2,
        lower_num_sample=10,
        upper_bound_sample=500,
        random_seed=42,
        y="Temperature",
        periodic_features=False,
        period=1440,
        spatial_features=True,
    ):
        df = pd.read_csv(file_path)
        self.y = y
        # Initialize the random seed
        np.random.seed(random_seed)
        if sampling_method == "chunk_by_sensor":
            self.train, self.val, self.test = self._chunk_by_sensor(
                df, lower_num_sample, upper_bound_sample, replicates
            )
        else:
            raise ValueError("The sampling method must be one of 'chunk_by_sensor'.")

        self.period = period  # minutes in a day
        self.periodic_features = periodic_features
        self.spatial_features = spatial_features

    def get_train(self):
        """
        Set up the covariates to be minutes, latitude, and longitude.
        """
        if self.spatial_features:
            X = self.train[["Minutes", "X", "Y"]].values
        else:
            X = self.train[["Minutes"]].values

        # If periodic features
        if self.periodic_features:
            X = self._periodize(X)

        y = self.train[self.y].values

        # Normalize y to have mean 0 and standard deviation 1
        y = (y - y.mean()) / y.std()
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

        if self.spatial_features:
            X = self.val[["Minutes", "X", "Y"]].values
        else:
            X = self.val[["Minutes"]].values

        y = self.val[self.y].values
        # X, y = self.val[["Minutes", "X", "Y"]].values, self.val[self.y].values

        # If periodic features
        if self.periodic_features:
            X = self._periodize(X)

        y_train = self.train[self.y].values
        y = (y - y_train.mean()) / y_train.std()

        return X, y

    def _chunk_by_sensor(self, df, lower_num_sample, upper_bound_sample, replicates):

        sensors = df["Sensor_ID"].unique()

        train_df = pd.DataFrame()
        val_df = pd.DataFrame()
        test_df = pd.DataFrame()

        df.sort_values("Minutes", inplace=True)
        val_samples = np.random.randint(
            lower_num_sample, upper_bound_sample, size=(len(sensors), replicates)
        )
        test_samples = np.random.randint(
            lower_num_sample, upper_bound_sample, size=(len(sensors), replicates)
        )

        for val_counts, test_counts, sensor in zip(val_samples, test_samples, sensors):

            sub_df = df[df["Sensor_ID"] == sensor]

            for rep_sample in val_counts:
                # Randomly select a starting point
                start = np.random.randint(0, len(sub_df) - rep_sample)
                end = start + rep_sample

                # Select the data from the data frame with the given indices
                # and remove from the data frame
                sample = sub_df.iloc[start:end]
                sub_df.drop(sample.index, inplace=True)

                val_df = pd.concat([val_df, sample])

            for rep_sample in test_counts:
                # Randomly select a starting point
                start = np.random.randint(0, len(sub_df) - rep_sample)
                end = start + rep_sample

                # Select the data from the data frame with the given indices
                # and remove from the data frame
                sample = sub_df.iloc[start:end]
                sub_df.drop(sample.index, inplace=True)

                test_df = pd.concat([test_df, sample])

            train_df = pd.concat([train_df, sub_df])

        assert len(train_df) + len(val_df) + len(test_df) == len(df)

        return train_df, val_df, test_df
