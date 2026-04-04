from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import TypedDict

import h5py
import numpy as np
import torch

__all__ = [
    "DataSet",
    "DataLoader",
    "DictDataSet",
    "LazyH5DataSet",
    "ModelInputDict",
]


class ModelInputDict(TypedDict):
    x: torch.Tensor
    cond: torch.Tensor
    num_points: torch.Tensor
    layer: torch.Tensor
    mask: torch.Tensor
    label: torch.Tensor
    noise: torch.Tensor | None


class DataSet(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index: int | list[int] | torch.Tensor) -> ModelInputDict:
        pass


class DataLoader(Iterable[ModelInputDict]):
    data_set: DataSet
    batch_size: int
    drop_last: bool
    shuffle: bool
    max_batch: int

    class __BatchIterator(Iterator[ModelInputDict]):
        def __init__(self, data_loader: "DataLoader", index: torch.Tensor) -> None:
            self.data_loader = data_loader
            self.batch = 0
            self.index = index

        def __next__(self) -> ModelInputDict:
            if self.batch >= self.data_loader.max_batch:
                raise StopIteration
            first = self.batch * self.data_loader.batch_size
            last = min(first + self.data_loader.batch_size, len(self.index))
            idx = self.index[first:last]
            self.batch += 1
            return self.data_loader.data_set[idx]

    def __init__(
        self,
        data_set: DataSet,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
    ) -> None:
        self.data_set = data_set
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        if self.drop_last:
            self.max_batch = len(self.data_set) // self.batch_size
        else:
            self.max_batch = (
                len(self.data_set) + self.batch_size - 1
            ) // self.batch_size

    def __len__(self) -> int:
        return self.max_batch

    def __iter__(self) -> Iterator[ModelInputDict]:
        if self.shuffle:
            index = torch.randperm(len(self.data_set))
        else:
            index = torch.arange(len(self.data_set))

        return self.__BatchIterator(self, index)


class DictDataSet(DataSet):
    def __init__(self, data: ModelInputDict) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data["x"])

    def __getitem__(self, index: int | list[int] | torch.Tensor) -> ModelInputDict:
        data = {}
        for key, value in self.data.items():
            if type(value) is torch.Tensor:
                data[key] = value[index].clone().detach()
            else:
                data[key] = None
        result = ModelInputDict(**data)
        return result


class LazyH5DataSet(DataSet):
    """Dataset that reads from an HDF5 file on-the-fly instead of loading
    everything into memory.  Designed for very large files (100+ GB) where
    the in-memory ``DictDataSet`` approach is infeasible.

    Preprocessing transformations are applied per-batch at read time.
    The H5 file is opened lazily on first access (and re-opened if the
    file handle has been closed, e.g. after pickling across workers).
    """

    def __init__(
        self,
        path: str,
        start: int,
        stop: int,
        *,
        label_map: dict[int, int] | None = None,
        samples_energy_trafo: "torch.nn.Module | None" = None,
        samples_coordinate_trafo: "torch.nn.Module | None" = None,
        cond_trafo: "torch.nn.Module | None" = None,
        samples_time_trafo: "torch.nn.Module | None" = None,
        return_noise: bool = False,
        return_direction: bool = False,
        max_num_points: int | None = None,
        num_layers: int = -1,
    ) -> None:
        self.path = path
        self.start = start
        self.stop = stop
        self.length = stop - start

        self.label_map = label_map
        self.samples_energy_trafo = samples_energy_trafo
        self.samples_coordinate_trafo = samples_coordinate_trafo
        self.cond_trafo = cond_trafo
        self.samples_time_trafo = samples_time_trafo
        self.return_noise = return_noise
        self.return_direction = return_direction
        self.max_num_points = max_num_points
        self.num_layers = num_layers

        self._file: h5py.File | None = None

    # -- file handle management ------------------------------------------

    @property
    def _h5(self) -> h5py.File:
        if self._file is None or not self._file.id.valid:
            self._file = h5py.File(self.path, "r")
        return self._file

    def close(self) -> None:
        if self._file is not None and self._file.id.valid:
            self._file.close()
            self._file = None

    # -- dataset interface -----------------------------------------------

    def __len__(self) -> int:
        return self.length

    def __del__(self) -> None:
        self.close()

    @torch.no_grad()
    def __getitem__(self, index: int | list[int] | torch.Tensor) -> ModelInputDict:
        if isinstance(index, torch.Tensor):
            index = index.numpy()
        if isinstance(index, np.ndarray):
            index = index.astype(np.int64)

        # Map dataset-relative indices to absolute file indices.
        abs_idx = np.asarray(index) + self.start

        # h5py fancy indexing requires sorted indices; we unsort after.
        sort_order = np.argsort(abs_idx)
        sorted_idx = abs_idx[sort_order]
        unsort_order = np.argsort(sort_order)

        f = self._h5
        showers_np = f["showers"][sorted_idx][unsort_order]
        energies_np = f["energies"][sorted_idx][unsort_order]
        pdg_np = f["pdg"][sorted_idx][unsort_order]

        if self.max_num_points is not None:
            showers_np = showers_np[:, : self.max_num_points, :]

        with_time = self.samples_time_trafo is not None
        if not with_time and showers_np.shape[2] == 5:
            showers_np = showers_np[:, :, :4]

        showers = torch.from_numpy(showers_np)
        energies = torch.from_numpy(energies_np)

        mask = showers[:, :, [3]] > 0
        energy = self.cond_trafo(energies) if self.cond_trafo is not None else energies

        if with_time:
            x = torch.cat(
                [
                    self.samples_coordinate_trafo(showers[:, :, :2])
                    if self.samples_coordinate_trafo is not None
                    else showers[:, :, :2],
                    self.samples_energy_trafo(showers[:, :, [3]])
                    if self.samples_energy_trafo is not None
                    else showers[:, :, [3]],
                    self.samples_time_trafo(showers[:, :, [4]])
                    if self.samples_time_trafo is not None
                    else showers[:, :, [4]],
                ],
                dim=-1,
            )
            x[~mask.repeat(1, 1, 4)] = 0.0
        else:
            x = torch.cat(
                [
                    self.samples_coordinate_trafo(showers[:, :, :2])
                    if self.samples_coordinate_trafo is not None
                    else showers[:, :, :2],
                    self.samples_energy_trafo(showers[:, :, [3]])
                    if self.samples_energy_trafo is not None
                    else showers[:, :, [3]],
                ],
                dim=-1,
            )
            x[~mask.repeat(1, 1, 3)] = 0.0

        layer = (showers[:, :, [2]] + 0.1).long()

        from allshowers.data_sets import batched_histogram, to_label_tensor

        num_points = batched_histogram(
            data=layer.squeeze(dim=-1),
            mask=mask.squeeze(dim=-1),
            num_bins=self.num_layers,
        )

        if self.label_map is not None:
            pdg_t = torch.from_numpy(pdg_np).view(-1)
            label = torch.zeros(pdg_t.shape[0], dtype=torch.int64)
            for idx_val, mapped in self.label_map.items():
                label[pdg_t == idx_val] = mapped
        else:
            label = to_label_tensor(torch.from_numpy(pdg_np))
            if label is None:
                label = torch.zeros(len(showers_np), dtype=torch.int64)

        if self.return_direction:
            directions_np = f["directions"][sorted_idx][unsort_order]
            directions = torch.from_numpy(directions_np)
            cond = torch.cat([energy, directions], dim=-1)
        else:
            cond = energy

        noise: torch.Tensor | None = None
        if self.return_noise and "target" in f:
            noise_np = f["target"][sorted_idx][unsort_order]
            target_pts = showers_np.shape[1]
            n_pts = noise_np.shape[1]
            if n_pts > target_pts:
                noise_np = noise_np[:, :target_pts, :]
            elif n_pts < target_pts:
                pad = np.zeros(
                    (noise_np.shape[0], target_pts - n_pts, noise_np.shape[2]),
                    dtype=noise_np.dtype,
                )
                noise_np = np.concatenate([noise_np, pad], axis=1)
            noise = torch.from_numpy(noise_np)

        return ModelInputDict(
            x=x,
            cond=cond,
            num_points=num_points,
            layer=layer,
            mask=mask,
            label=label,
            noise=noise,
        )
