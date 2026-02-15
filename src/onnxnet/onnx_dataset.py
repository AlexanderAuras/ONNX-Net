import csv
from pathlib import Path
import pickle  # noqa: S403
from typing import Any, override

import numpy as np
from torch.utils.data import Dataset


class ONNXDataset(Dataset[dict[str, Any]]):
    def __init__(self, path: Path | str) -> None:
        self.__path = Path(path)
        if not self.__path.exists() or not self.__path.is_dir():
            msg = f'Invalid dataset directory "{self.__path}"'
            raise FileNotFoundError(msg)
        self.__metadata = []
        with self.__path.joinpath("metadata.csv").open("r", encoding="UTF-8") as f:
            csv_reader = csv.reader(f)
            next(csv_reader)  # skip header
            for row in csv_reader:
                self.__metadata.append({"file": row[0], "accuracy": row[1], "tokens": row[2], "dataset": row[3]})

    def __len__(self) -> int:
        return len(self.__metadata)

    @override
    def __getitem__(self, idx: int) -> dict[str, Any]:  # ty: ignore [invalid-method-override]
        metadata = self.__metadata[idx]
        rel_path = Path(metadata["file"])
        accuracy = metadata["accuracy"]
        num_tokens = metadata["tokens"]
        dataset = metadata["dataset"]
        with self.__path.joinpath(rel_path).with_suffix(".tkid").open("rb") as f:
            tknzr_outp = pickle.load(f)  # noqa: S301
        with self.__path.joinpath(rel_path).with_suffix(".ndid").open("rb") as f:
            node_ids = pickle.load(f)  # noqa: S301
        pos_enc = np.load(self.__path.joinpath(rel_path).with_suffix(".npy"))
        return {
            **tknzr_outp,
            "node_ids": node_ids,
            "positional_encodings": pos_enc,
            "accuracy": accuracy,
            "num_tokens": num_tokens,
            "dataset": dataset,
        }
