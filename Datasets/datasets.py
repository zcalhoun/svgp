import os
import rasterio as rio
from numpy import ma
import numpy as np
from typing import Type, Tuple
from copy import deepcopy


class Dataset:
    def __init__(self, data) -> None:
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Observation:
    """This class is used as a wrapper around the masked array data."""

    def __init__(self, data: ma.array) -> None:
        self.data = data
        self.indices = np.stack(np.where(~data.mask), axis=-1)
        self.shape = data.shape

    def __len__(self) -> int:
        return np.sum(~self.data.mask)


class SpatialDataset:

    def __init__(
        self,
        n: int = 24,
        path: str = "../data/half_km_res/temp",
    ) -> None:
        self.files = os.listdir(path)
        self.files.sort()
        self.files = self.files[:n]
        self.path = path

        self.coords = self._create_mask()
        self.shape = self.__getitem__(0, obs_only=False).shape

    def __len__(self) -> int:
        return len(self.files)

    def _create_mask(self) -> Tuple[Type[np.ndarray], Type[np.ndarray]]:
        shape = self.__getitem__(0, obs_only=False).shape
        mask = np.zeros(dtype=bool, shape=shape)
        for i in range(len(self)):
            d = self.__getitem__(i, obs_only=False)
            mask = mask | ~np.isnan(d)
        idx = np.where(mask)
        indices = np.stack(idx, axis=-1)
        return indices

    def __getitem__(self, idx, obs_only=True) -> np.ndarray:
        with rio.open(os.path.join(self.path, self.files[idx])) as src:
            data = src.read(1)

        if obs_only:
            return data[self.coords[:, 0], self.coords[:, 1]]
        else:
            return data

    def train_test_split(
        self, hold_out: int = 0.4, random_seed: int = 42
    ) -> Tuple[Type["July2023"], Type["July2023"]]:
        """
        This function splits the dataset into a train/test split
        so that we can validate the model on the dataset in a reasonable way,
        so that we do not access the same points in the training and test set.

        Args:
            hold_out (int) : The fraction of the dataset to hold out for testing.
            random_seed (int) : The random seed to use for the split.
        """
        np.random.seed(random_seed)

        test_idx = np.random.binomial(1, hold_out, size=len(self.coords))

        train_set = deepcopy(self)
        train_set.coords = self.coords[~(test_idx == 1)]

        test_set = deepcopy(self)
        test_set.coords = self.coords[test_idx == 1]

        return train_set, test_set


class July2023:
    def __init__(
        self,
        hold_out: float = 0.4,
        n: int = 24,
        path: str = "../data/half_km_res/temp",
        random_seed: int = 42,
    ) -> None:
        """
        This class allows us to access the first n hours of the July 2023 dataset.
        """
        self.files = os.listdir(path)
        self.files.sort()
        self.files = self.files[:n]
        self.path = path
        self.is_train = None

        # self.mu = self._calc_mu()
        # self.std = self._calc_std()
        self.mu = []
        self.std = []
        self.mask, self.indices = self._create_mask()
        # self.mu = self._calc_mu()
        # self.std = self._calc_std()

        self.shape = self.__getitem__(0, normalize=False).shape

    def _create_mask(self) -> Tuple[Type[np.ndarray], Type[np.ndarray]]:
        shape = self.__getitem__(0, normalize=False).shape
        mask = np.zeros(dtype=bool, shape=shape)
        for i in range(len(self)):
            d = self.__getitem__(i, normalize=False)
            mask = mask | ~np.isnan(d)
        mask = ~mask.mask
        idx = np.where(mask)
        indices = np.stack(idx, axis=-1)
        return mask, indices

    def __len__(self) -> int:
        return len(self.files)

    def _calc_mu(self) -> np.ndarray:
        mu = np.zeros(len(self))
        for i in range(len(self)):
            d = self.__getitem__(i, normalize=False)
            mu[i] = np.nanmean(d)
        return mu

    def _calc_std(self) -> np.ndarray:
        std = np.zeros(len(self))
        for i in range(len(self)):
            d = self.__getitem__(i, normalize=False)
            std[i] = np.nanstd(d)
        return std

    def __getitem__(self, idx, normalize=True) -> Type[Observation]:
        with rio.open(os.path.join(self.path, self.files[idx])) as src:
            data = src.read(1)

        data = ma.array(data, mask=np.isnan(data))
        # mu = np.mean(data)
        # sigma = np.std(data)
        # data = (data - mu) / sigma

        if self.is_train is not None:
            mask = np.zeros(dtype=bool, shape=data.shape)
            mask[self.indices[:, 0], self.indices[:, 1]] = True
            mask = np.isnan(data) | ~mask
            data = ma.array(data, mask=mask)

        if normalize:
            data = (data - self.mu[idx]) / self.std[idx]
        return data

    def train_test_split(
        self, hold_out: int = 0.4, random_seed: int = 42
    ) -> Tuple[Type["July2023"], Type["July2023"]]:
        """
        This function splits the dataset into a train/test split
        so that we can validate the model on the dataset in a reasonable way,
        so that we do not access the same points in the training and test set.

        Args:
            hold_out (int) : The fraction of the dataset to hold out for testing.
            random_seed (int) : The random seed to use for the split.
        """
        np.random.seed(random_seed)

        test_idx = np.random.binomial(1, hold_out, size=len(self.indices))

        train_set = deepcopy(self)
        train_set.indices = self.indices[~(test_idx == 1)]
        train_set.is_train = True
        train_set.mu = train_set._calc_mu()
        train_set.std = train_set._calc_std()

        test_set = deepcopy(self)
        test_set.indices = self.indices[test_idx == 1]
        test_set.is_train = False
        test_set.mu = train_set.mu
        test_set.std = train_set.std

        return train_set, test_set
