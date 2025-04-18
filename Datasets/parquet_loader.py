import pandas as pd
import glob
import random
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
from typing import List


def train_test_split(root_dir, random_seed=42, train_size=0.8):
    """
    Splits the dataset into train and test sets based on the station partitions.

    Parameters:
        root_dir (str): Root directory where station=*/year=*/month=* partitions live.
        random_seed (int): Random seed for reproducibility.
        train_size (float): Proportion of the dataset to include in the train split.

    Returns:
        tuple: Two lists of paths for train and test sets.
    """
    random.seed(random_seed)
    all_stations = glob.glob(os.path.join(root_dir, "station=*"))
    random.shuffle(all_stations)

    split_index = int(len(all_stations) * train_size)
    train_stations = all_stations[:split_index]
    test_stations = all_stations[split_index:]

    return train_stations, test_stations


class ParquetStationBatchLoader:
    def __init__(
        self,
        station_paths: List[str],
        features: List[str],
        target: List[str],
        file_chunk_size: int = 100,
        batch_size: int = 1024,
        shuffle_rows: bool = True,
        shuffle_paths: bool = True,
        device: str = "cpu",
    ):
        """
        Parameters:
            station_paths (str): Station path names.
            features (List[str]): List of feature column names.
            target (List[str]): List of target column names.
            batch_size (int): Batch size for DataLoader.
            shuffle_paths (bool): Whether to shuffle paths within each station.
            device (str): Device to load tensors onto ("cpu" or "cuda").
        """
        self.station_paths = station_paths
        self.features = features
        self.target = target
        self.batch_size = batch_size
        self.file_chunk_size = file_chunk_size
        self.shuffle_paths = shuffle_paths
        self.shuffle_rows = shuffle_rows
        self.device = device

        self._all_paths = self._create_all_paths()

    def _create_all_paths(self):

        all_paths = []
        for station_path in self.station_paths:
            year_month_paths = glob.glob(
                os.path.join(station_path, "year=*/month=*/*.parquet")
            )
            all_paths.extend(year_month_paths)

        return all_paths

    def __iter__(self):
        all_paths = self._all_paths.copy()

        if self.shuffle_paths:
            random.shuffle(all_paths)

        for i in range(0, len(all_paths), self.file_chunk_size):
            file_paths = all_paths[i : i + self.file_chunk_size]

            dfs = [pd.read_parquet(p) for p in file_paths]
            df = pd.concat(dfs, ignore_index=True)

            if self.shuffle_rows:
                df = df.sample(frac=1).reset_index(drop=True)

            X = torch.tensor(df[self.features].values, dtype=torch.float32).to(
                self.device
            )
            y = torch.tensor(df[self.target].values, dtype=torch.float32).to(
                self.device
            )

            dataset = TensorDataset(X, y)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            yield loader
