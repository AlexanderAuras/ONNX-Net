from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast, override

from torch.utils.data import Dataset


if TYPE_CHECKING:
    from collections.abc import Sized


class TransformDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        dataset: Dataset[dict[str, Any]],
        *,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.__dataset = dataset
        self.__transform = transform if transform is not None else lambda x: x

    def __len__(self) -> int:
        return len(cast("Sized", self.__dataset))

    @override
    def __getitem__(self, idx: int) -> dict[str, Any]:  # ty: ignore [invalid-method-override]
        return self.__transform(self.__dataset[idx])
