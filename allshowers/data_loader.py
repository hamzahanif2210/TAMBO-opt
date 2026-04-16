from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import TypedDict

import torch

__all__ = ["DataSet", "DataLoader", "DictDataSet", "ModelInputDict"]


class ModelInputDict(TypedDict):
    x: torch.Tensor
    cond: torch.Tensor
    num_points: torch.Tensor
    layer: torch.Tensor
    z_depth: torch.Tensor  # continuous z position in metres (per point)
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
