import os
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
    "ShardedDataLoader",
    "ShardedDataSet",
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


class ShardedDataSet(DataSet):
    """Dataset backed by pre-transformed ``.pt`` shard files.

    Only one shard is kept in memory at a time.  At the start of each
    epoch the ``DataLoader`` calls ``__iter__`` which generates a fresh
    permutation; batches that fall within the current shard are served
    from RAM, and the shard is swapped when the next batch needs a
    different one.

    Because random global shuffling would require swapping shards on
    every batch, this dataset uses a *shard-then-shuffle* strategy:

    1. Shuffle the order of shards each epoch.
    2. Within each shard, shuffle the sample order.

    This gives good randomness while keeping I/O sequential.
    """

    def __init__(self, shard_dir: str, prefix: str) -> None:
        self.shard_dir = shard_dir
        self.prefix = prefix

        # Discover shard files
        self.shard_files: list[str] = sorted(
            f
            for f in os.listdir(shard_dir)
            if f.startswith(prefix + "_") and f.endswith(".pt")
        )
        if not self.shard_files:
            raise FileNotFoundError(
                f"No shard files matching '{prefix}_*.pt' in {shard_dir}"
            )

        # Compute total length and per-shard lengths
        self.shard_lengths: list[int] = []
        for sf in self.shard_files:
            data = torch.load(os.path.join(shard_dir, sf), weights_only=False)
            self.shard_lengths.append(len(data["x"]))
            del data
        self.total_length = sum(self.shard_lengths)

        # Current loaded shard
        self._loaded_shard_idx: int = -1
        self._loaded_data: dict | None = None

    def __len__(self) -> int:
        return self.total_length

    def _load_shard(self, shard_idx: int) -> dict:
        if shard_idx != self._loaded_shard_idx:
            # Free previous shard
            self._loaded_data = None
            path = os.path.join(self.shard_dir, self.shard_files[shard_idx])
            self._loaded_data = torch.load(path, weights_only=False)
            self._loaded_shard_idx = shard_idx
        return self._loaded_data  # type: ignore[return-value]

    def _global_to_shard(self, global_idx: int) -> tuple[int, int]:
        """Map a global index to (shard_index, local_index)."""
        offset = 0
        for i, length in enumerate(self.shard_lengths):
            if global_idx < offset + length:
                return i, global_idx - offset
            offset += length
        raise IndexError(f"Index {global_idx} out of range [0, {self.total_length})")

    def __getitem__(self, index: int | list[int] | torch.Tensor) -> ModelInputDict:
        if isinstance(index, (int, np.integer)):
            index = torch.tensor([index])
        if isinstance(index, list):
            index = torch.tensor(index)

        # Group indices by shard for efficient loading
        shard_indices: dict[int, list[tuple[int, int]]] = {}
        for batch_pos, global_idx in enumerate(index.tolist()):
            shard_idx, local_idx = self._global_to_shard(global_idx)
            shard_indices.setdefault(shard_idx, []).append((batch_pos, local_idx))

        batch_size = len(index)
        result_parts: dict[str, list] = {}

        # Allocate in shard order to minimise shard swaps
        for shard_idx in sorted(shard_indices.keys()):
            pairs = shard_indices[shard_idx]
            data = self._load_shard(shard_idx)
            local_idxs = torch.tensor([p[1] for p in pairs])
            batch_positions = [p[0] for p in pairs]

            for key, value in data.items():
                if key not in result_parts:
                    result_parts[key] = [None] * batch_size
                if isinstance(value, torch.Tensor):
                    sliced = value[local_idxs]
                    for bp, row in zip(batch_positions, sliced):
                        result_parts[key][bp] = row
                else:
                    # None (e.g. noise)
                    for bp in batch_positions:
                        result_parts[key][bp] = value

        # Stack into tensors
        out: dict = {}
        for key, parts in result_parts.items():
            if parts[0] is None:
                out[key] = None
            else:
                out[key] = torch.stack(parts)
        return ModelInputDict(**out)


class ShardedDataLoader(Iterable[ModelInputDict]):
    """DataLoader optimised for ``ShardedDataSet``.

    Instead of globally shuffling indices (which causes thrashing between
    shards), this loader:

    1. Shuffles the **shard order** each epoch.
    2. Loads one shard at a time into a ``DictDataSet``.
    3. Shuffles **within** that shard and yields batches.

    This means each shard is loaded exactly once per epoch, and the only
    RAM used is one shard at a time.
    """

    def __init__(
        self,
        data_set: ShardedDataSet,
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
        num_shards = len(self.data_set.shard_files)
        if self.shuffle:
            shard_order = torch.randperm(num_shards).tolist()
        else:
            shard_order = list(range(num_shards))

        batches_yielded = 0
        for shard_idx in shard_order:
            shard_data = self.data_set._load_shard(shard_idx)
            shard_len = self.data_set.shard_lengths[shard_idx]

            if self.shuffle:
                perm = torch.randperm(shard_len)
            else:
                perm = torch.arange(shard_len)

            for start in range(0, shard_len, self.batch_size):
                end = min(start + self.batch_size, shard_len)
                if self.drop_last and (end - start) < self.batch_size:
                    continue
                idx = perm[start:end]

                batch: dict = {}
                for key, value in shard_data.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value[idx].clone().detach()
                    else:
                        batch[key] = None

                yield ModelInputDict(**batch)
                batches_yielded += 1

        # Pad with partial shard data is already handled by drop_last


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
