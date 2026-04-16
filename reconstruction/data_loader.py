"""
Data loader for the reconstruction model.

Loads a combined HDF5 file (produced by util/dataset_for_reconstruction.py)
and builds:
  - data   : directions (3) concatenated with label (1)  -> shape (N, 4)
  - condition : selected per-particle-type layer observables  -> shape (N, D_cond)

The ``condition_features`` config list controls which per-particle-type
observables are included and in what order.  Each entry is a dataset name
suffix; the loader will look for ``<suffix>_electron``, ``<suffix>_muon``,
``<suffix>_photon`` in the HDF5 file.

Example ``condition_features``:
  - num_points_per_layer
  - energy_per_layer
  - time_per_layer

This would yield D_cond = 3 * 3 * num_layers  (3 features x 3 particles x L).
"""

import os

import h5py
import numpy as np
import torch
from torch import Tensor

from reconstruction.preprocessing import Identity, Transformation, compose


PARTICLE_TYPES = ["electron", "muon", "photon"]

# Default list of per-layer observable keys (without particle suffix)
DEFAULT_CONDITION_FEATURES = [
    "num_points_per_layer",
    "energy_per_layer",
    "time_per_layer",
]


def load_dataset(
    file: h5py.File,
    key: str,
    start: int = 0,
    end: int | None = None,
) -> Tensor:
    if key not in file:
        raise KeyError(f"Key '{key}' not found in HDF5 file.")
    dataset = file[key]
    if not isinstance(dataset, h5py.Dataset):
        raise TypeError(f"Key '{key}' is not a dataset in HDF5 file.")
    data = dataset[start:end]
    is_integer = issubclass(data.dtype.type, (np.integer, np.bool_))
    tensor_dtype = torch.int64 if is_integer else torch.get_default_dtype()
    return torch.from_numpy(data).to(tensor_dtype, copy=False)


def load_data_file(
    data_file: str,
    condition_features: list[str] | None = None,
    start: int = 0,
    end: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Load reconstruction data from a single HDF5 file.

    Returns:
        (target, condition) where
        target    = [directions(3), label(1)]   shape (N, 4)
        condition = concatenated per-particle-type observables  shape (N, D)
    """
    if condition_features is None:
        condition_features = DEFAULT_CONDITION_FEATURES

    with h5py.File(data_file, "r") as file:
        directions = load_dataset(file, "directions", start, end)  # (N, 3)
        labels = load_dataset(file, "labels", start, end)          # (N,)

        cond_parts: list[Tensor] = []
        for feat in condition_features:
            for ptype in PARTICLE_TYPES:
                key = f"{feat}_{ptype}"
                cond_parts.append(load_dataset(file, key, start, end))

    # Target: directions + label (as float for flow matching)
    labels_float = labels.to(torch.get_default_dtype()).unsqueeze(-1)
    target = torch.cat([directions, labels_float], dim=-1)  # (N, 4)

    condition = torch.cat(cond_parts, dim=-1)  # (N, D_cond)

    return target, condition


class DataLoader:
    def __init__(
        self,
        data_file: str,
        condition_features: list[str] | None = None,
        transform_data: Transformation | list | None = None,
        transform_condition: Transformation | list | None = None,
        batch_size: int = 1,
        shuffle: bool = False,
        start: int = 0,
        end: int | None = None,
        fit_transform: bool = False,
        device: torch.device | str = "cpu",
    ) -> None:
        self.data_file = data_file
        self.transform_data = self.__compose_trafo(transform_data)
        self.transform_condition = self.__compose_trafo(transform_condition)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.condition_features = condition_features or DEFAULT_CONDITION_FEATURES

        target, condition = load_data_file(
            data_file,
            condition_features=self.condition_features,
            start=start,
            end=end,
        )

        self.num_samples = target.shape[0]
        target = target.to(device)
        condition = condition.to(device)

        if fit_transform:
            target = self.transform_data.fit(target)
            condition = self.transform_condition.fit(condition)
        else:
            target = self.transform_data(target)
            condition = self.transform_condition(condition)

        self.data = target          # (N, 4)  directions + pdg label
        self.condition = condition   # (N, D_cond)

    @staticmethod
    def __compose_trafo(transformation: Transformation | list | None) -> Transformation:
        if transformation is None:
            return Identity()
        if isinstance(transformation, list):
            return compose(transformation)
        return transformation

    def __len__(self) -> int:
        return self.num_samples // self.batch_size

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.num_samples)
        else:
            indices = torch.arange(self.num_samples)
        for i in range(len(self)):
            idx = indices[i * self.batch_size : (i + 1) * self.batch_size]
            yield {
                "data": self.data[idx],
                "condition": self.condition[idx],
                "noise": None,
            }

    def to(self, device_dtype: torch.device | torch.dtype | str) -> None:
        self.data = self.data.to(device_dtype)
        self.condition = self.condition.to(device_dtype)
        self.transform_data.to(device_dtype)
        self.transform_condition.to(device_dtype)


def get_loaders(
    data_file: str,
    condition_features: list[str] | None = None,
    transform_data: Transformation | list | None = None,
    transform_condition: Transformation | list | None = None,
    batch_size: int = 128,
    batch_size_val: int | None = None,
    device: torch.device | str = "cpu",
    num_train: int | None = None,
    num_val: int = 10_000,
) -> tuple[DataLoader, DataLoader]:
    if num_train is None:
        num_train = -num_val
    if batch_size_val is None:
        batch_size_val = batch_size
    train_loader = DataLoader(
        data_file,
        condition_features=condition_features,
        transform_data=transform_data,
        transform_condition=transform_condition,
        batch_size=batch_size,
        shuffle=True,
        start=0,
        end=num_train,
        fit_transform=True,
        device=device,
    )
    val_loader = DataLoader(
        data_file,
        condition_features=condition_features,
        transform_data=train_loader.transform_data,
        transform_condition=train_loader.transform_condition,
        batch_size=batch_size_val,
        shuffle=False,
        start=-num_val,
        end=None,
        fit_transform=False,
        device=device,
    )
    return train_loader, val_loader
