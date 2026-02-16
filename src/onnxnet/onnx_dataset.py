import csv
from pathlib import Path
import pickle  # noqa: S403
from typing import Any, override

import numpy as np
from torch.utils.data import Dataset


class ONNXDataset(Dataset[dict[str, Any]]):
    def __init__(self, path: Path | str) -> None:
        path = Path(path)
        self.__path = path.parent
        if not path.exists() or not path.is_file():
            msg = f'Invalid dataset file "{path}"'
            raise FileNotFoundError(msg)
        self.__metadata = []
        with path.open("r", encoding="UTF-8") as f:
            csv_reader = csv.reader(f)
            next(csv_reader)  # skip header
            for row in csv_reader:
                self.__metadata.append({
                    "file": self.__path.joinpath(row[0]),
                    "accuracy": float(row[1]),
                    "dataset": row[2],
                })

    def __len__(self) -> int:
        return len(self.__metadata)

    @override
    def __getitem__(self, idx: int) -> dict[str, Any]:  # ty: ignore [invalid-method-override]
        metadata = self.__metadata[idx]
        path = metadata["file"]
        accuracy = metadata["accuracy"]
        dataset = metadata["dataset"]
        model_str = path.with_suffix(".mstr").read_text()
        with path.with_suffix(".chid").open("rb") as f:
            char_ids = pickle.load(f)  # noqa: S301
        pos_enc = np.load(path.with_suffix(".npy"))
        return {
            "model_str": model_str,
            "char_ids": char_ids,
            "positional_encodings": pos_enc,
            "accuracy": accuracy,
            "dataset": dataset,
        }
