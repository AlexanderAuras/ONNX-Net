from typing import Any

from transformers import DataCollatorWithPadding


class CustomPECollator(DataCollatorWithPadding):
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        token_pes = [f["token_pes"] for f in features]
        max_len = max(len(t) for t in token_pes)
        hidden_dim = len(features[0]["token_pes"][0])
        padded = []
        for t in token_pes:
            pad_len = max_len - len(t)
            padded.append(t + [[0] * hidden_dim] * pad_len)
        for i, f in enumerate(features):
            f["token_pes"] = padded[i]
        batch = super().__call__(features)
        return batch
