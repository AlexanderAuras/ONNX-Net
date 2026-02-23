from __future__ import annotations

from pathlib import Path
import pickle  # noqa: S403
from typing import TYPE_CHECKING, Any, Self, cast, override

import lmdb
import torch
from torch.utils.data import Dataset


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sized
    from types import TracebackType


class ImprovedDataset(Dataset[dict[str, Any]]):
    __FLOAT_SUM_CHECK_TOLERANCE = 1e-5

    def __init__(
        self,
        dataset: Dataset[dict[str, Any]],
        indices: Iterable[int] | None = None,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self._dataset = dataset
        self._indices = list(set(indices)) if indices is not None else None
        if self._indices is not None and (
            min(self._indices) < 0 or max(self._indices) >= len(cast("Sized", self._dataset))
        ):
            msg = "Index out of bounds"
            raise ValueError(msg)
        self._transform = transform if transform is not None else lambda x: x

    def __len__(self) -> int:
        if self._indices is not None:
            return len(self._indices)
        return len(cast("Sized", self._dataset))

    @override
    def __getitem__(self, idx: int) -> dict[str, Any]:  # ty: ignore [invalid-method-override]
        if idx < 0 or idx >= len(self):
            raise IndexError
        if self._indices is not None:
            return self._transform(self._dataset[self._indices[idx]])
        return self._transform(self._dataset[idx])

    def subset(self, indices: Iterable[int]) -> ImprovedDataset:
        indices = list(indices)
        if min(indices) < 0 or max(indices) >= len(self):
            msg = "Index out of bounds"
            raise ValueError(msg)
        if self._indices is not None:
            indices = [self._indices[i] for i in indices]
        dataset = ImprovedDataset(self._dataset, indices, self._transform)
        return dataset

    def random_split(
        self,
        *args: float,
        generator: torch.Generator | None = None,
    ) -> tuple[ImprovedDataset, ...]:
        if abs(1.0 - sum(args)) > ImprovedDataset.__FLOAT_SUM_CHECK_TOLERANCE:
            msg = "Fractions must add up to 1"
            raise ValueError(msg)
        all_indices = torch.randperm(len(self), generator=generator).tolist()
        counts = [round(len(self) * x) for x in args[:-1]]
        counts += [len(self) - sum(counts)]
        result = []
        for count in counts:
            indices = all_indices[:count]
            all_indices = all_indices[count:]
            if self._indices is not None:
                indices = [self._indices[i] for i in indices]
            result.append(ImprovedDataset(self._dataset, indices, self._transform))
        return tuple(result)

    def map(self, transform: Callable[[dict[str, Any]], dict[str, Any]]) -> ImprovedDataset:
        dataset = ImprovedDataset(self._dataset, self._indices, lambda x: transform(self._transform(x)))
        return dataset


class LMDBDataset(ImprovedDataset):
    class __LMDBDataset(Dataset[dict[str, Any]]):
        def __init__(
            self,
            path: str | Path,
            *,
            map_size: int = 2**40,
        ) -> None:
            super().__init__()
            self.__path = Path(path).with_suffix(".lmdb")
            self.__current_file_no = None
            self.__env = lmdb.open(str(self.__path), create=False, readonly=True, subdir=False, map_size=map_size)
            self.__length = self.__env.stat()["entries"]
            self.__closed = False

        @property
        def path(self) -> Path:
            return self.__path

        def __len__(self) -> int:
            return self.__length

        @override
        def __getitem__(self, idx: int) -> dict[str, Any]:  # ty: ignore [invalid-method-override]
            if self.__closed:
                raise ValueError
            with self.__env.begin() as txn:
                sample = pickle.loads(txn.get(idx.to_bytes(4, "big", signed=False)))  # noqa: S301
            return sample

        def __del__(self) -> None:
            self.__env.close()

        def close(self) -> None:
            self.__env.close()
            self.__closed = True

    def __init__(
        self,
        path: str | Path,
        *,
        map_size: int = 2**40,
        indices: Iterable[int] | None = None,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(LMDBDataset.__LMDBDataset(path, map_size=map_size), indices, transform)

    @property
    def path(self) -> Path:
        return cast("LMDBDataset.__LMDBDataset", self._dataset).path

    def close(self) -> None:
        cast("LMDBDataset.__LMDBDataset", self._dataset).close()


class LMDBDatasetWriter:
    def __init__(self, path: str | Path, *, map_size: int = 2**40) -> None:
        self.__path = Path(path).with_suffix(".lmdb")
        self.__env: lmdb.Environment | None = None  # ty: ignore [possibly-missing-attribute]
        self.__map_size = map_size

    def __enter__(self) -> Self:
        self.__env = cast(
            "lmdb.Environment",  # ty: ignore [possibly-missing-attribute]
            lmdb.open(
                str(self.__path.parent.joinpath(f"{self.__path.name}")),
                create=True,
                subdir=False,
                map_size=self.__map_size,
            ),
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self.__env is None:
            return
        self.__env.close()

    def add(self, *args: dict[str, Any]) -> None:
        if self.__env is None:
            msg = "LMDBDatasetWriter must be used as a context manager"
            raise RuntimeError(msg)
        with self.__env.begin(write=True) as txn:
            next_index = self.__env.stat()["entries"]
            for sample in args:
                data = pickle.dumps(sample)
                txn.put(next_index.to_bytes(4, "big", signed=False), data)
                next_index += 1
