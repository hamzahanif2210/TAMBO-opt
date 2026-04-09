from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from typing import TypedDict

import torch

__all__ = [
    "DataSet",
    "DataLoader",
    "DictDataSet",
    "ChunkedDataLoader",
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


class ChunkedDataLoader(Iterable[ModelInputDict]):
    """DataLoader that streams data from disk in chunks to handle datasets
    that don't fit in memory.

    Instead of loading the full dataset at once, it:
    1. Divides the data range [start, stop) into chunks of ``chunk_size`` samples
    2. Each epoch, shuffles the chunk order and iterates through all chunks
    3. For each chunk, loads it from disk via ``load_fn``, serves batches, then
       frees the memory before loading the next chunk

    This keeps peak memory usage proportional to ``chunk_size`` rather than
    the full dataset size.
    """

    def __init__(
        self,
        load_fn: Callable[[int, int], ModelInputDict],
        total_samples: int,
        chunk_size: int,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
    ) -> None:
        self.load_fn = load_fn
        self.total_samples = total_samples
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        # Pre-compute chunk boundaries
        self.chunk_starts = list(range(0, total_samples, chunk_size))

        # Total number of batches across all chunks (approximate for __len__)
        if drop_last:
            self._total_batches = sum(
                min(chunk_size, total_samples - s) // batch_size
                for s in self.chunk_starts
            )
        else:
            self._total_batches = sum(
                (min(chunk_size, total_samples - s) + batch_size - 1) // batch_size
                for s in self.chunk_starts
            )

    def __len__(self) -> int:
        return self._total_batches

    class _ChunkedIterator(Iterator[ModelInputDict]):
        def __init__(self, loader: "ChunkedDataLoader") -> None:
            self.loader = loader
            # Shuffle chunk order each epoch
            if loader.shuffle:
                self.chunk_order = torch.randperm(len(loader.chunk_starts)).tolist()
            else:
                self.chunk_order = list(range(len(loader.chunk_starts)))
            self.chunk_idx = 0
            self.current_data: DictDataSet | None = None
            self.batch_idx = 0
            self.sample_indices: torch.Tensor | None = None
            self.max_batch_in_chunk = 0

        def _load_next_chunk(self) -> bool:
            """Load the next chunk from disk. Returns False if no chunks left."""
            if self.chunk_idx >= len(self.chunk_order):
                return False

            ci = self.chunk_order[self.chunk_idx]
            start = self.loader.chunk_starts[ci]
            stop = min(start + self.loader.chunk_size, self.loader.total_samples)
            self.chunk_idx += 1

            data = self.loader.load_fn(start, stop)
            self.current_data = DictDataSet(data)

            chunk_len = len(self.current_data)
            if self.loader.shuffle:
                self.sample_indices = torch.randperm(chunk_len)
            else:
                self.sample_indices = torch.arange(chunk_len)

            if self.loader.drop_last:
                self.max_batch_in_chunk = chunk_len // self.loader.batch_size
            else:
                self.max_batch_in_chunk = (
                    chunk_len + self.loader.batch_size - 1
                ) // self.loader.batch_size
            self.batch_idx = 0
            return True

        def __next__(self) -> ModelInputDict:
            # If no chunk loaded yet or current chunk exhausted, load next
            while (
                self.current_data is None
                or self.batch_idx >= self.max_batch_in_chunk
            ):
                # Free previous chunk
                self.current_data = None
                if not self._load_next_chunk():
                    raise StopIteration

            first = self.batch_idx * self.loader.batch_size
            last = min(
                first + self.loader.batch_size, len(self.sample_indices)
            )
            idx = self.sample_indices[first:last]
            self.batch_idx += 1
            return self.current_data[idx]

    def __iter__(self) -> Iterator[ModelInputDict]:
        return self._ChunkedIterator(self)
