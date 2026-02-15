import argparse
import json
from pathlib import Path

import pandas as pd
from scipy import stats


parser = argparse.ArgumentParser()
parser.add_argument("--res_path", type=str)
args = parser.parse_args()

df = pd.read_csv(Path(args.res_path) / "preds.csv")
output_json = Path(args.res_path) / "results.json"

res = {}
datasets = df["dataset"].unique()
for dataset in datasets:
    subset = df[df["dataset"] == dataset]
    spearman = stats.spearmanr(subset["pred"], subset["true"])[0]
    kendall = stats.kendalltau(subset["pred"], subset["true"])[0]
    res[dataset] = {
        "spearman": spearman,
        "kendall": kendall,
    }

with Path(output_json).open("w", encoding="UTF-8") as f:
    json.dump(res, f, indent=4)
