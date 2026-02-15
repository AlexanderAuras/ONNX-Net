import argparse
import datetime
import json
from pathlib import Path
from typing import Any, cast

from accelerate import Accelerator
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.trainer import Trainer
from transformers.trainer_utils import set_seed
from transformers.training_args import TrainingArguments

from onnxnet.losses import (
    HuberTrainer,
    MSECapTrainer,
    PairwiseLogTrainer,
    Poly1ListwiseTrainer,
    # SpearmanTrainer,
    PWRMiningTrainer,
    # PlackettTrainer,
    # ApproxRankTrainer,
    PWRTrainer,
    SoftmaxListwiseTrainer,
)
from onnxnet.onnx_dataset import ONNXDataset
import wandb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

# Model params
parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
parser.add_argument("--adv-pos-enc", action="store_true")

# Data params
parser.add_argument("--data_path", type=Path)
parser.add_argument("--eval_path", type=Path, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--train_task",
    type=str,
    default="none",
    choices=["einspace", "hnasbench201", "nasbench101", "nas201nats", "nasbench301", "none"],
)
parser.add_argument(
    "--eval_task",
    type=str,
    default="none",
    choices=["einspace", "hnasbench201", "nasbench101", "nas201nats", "nasbench301", "none"],
)
parser.add_argument("--train_size", type=int, default=None)

# Train params
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--warmup_ratio", type=float, default=0.06)
parser.add_argument("--wandb_project", type=str, default="eintool_surrogate")
parser.add_argument(
    "--loss_fn",
    type=str,
    default="mse",
    choices=[
        "mse",
        "huber",
        "mse_cap",
        "pwr",
        "spearman",
        "plackett",
        "approxrank",
        "pwr_mining",
        "pairlog",
        "softmaxlist",
        "poly1list",
    ],
)
parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["steps", "epoch", "no"])
parser.add_argument("--flash_attention", type=bool, default=False)
parser.add_argument("--gradient_checkpointing", type=bool, default=False)


# Output params
parser.add_argument("--output_path", type=Path, default=None)
parser.add_argument("--save", type=bool, default=True)
parser.add_argument("--save_pred", type=bool, default=True)

args = parser.parse_args()

data_name = args.data_path.name
run_name = f"eval_task{args.eval_task}_train_task{args.train_task}_{args.loss_fn}_{data_name}\
             _seed{args.seed}_lr{args.lr}_epochs{args.epochs}_frac42"

accelerator = Accelerator()
if accelerator.is_main_process:
    _ = wandb.init(
        project=args.wandb_project,
        config=vars(args),
        name=run_name,
        mode="disabled",  # TODO Change back
    )

# Reproducibility
set_seed(args.seed, deterministic=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# if args.train_task != "none" and args.eval_task != "none":
#    msg = "Train task and eval task cannot be set at the same time"
#    raise ValueError(msg)

if not args.data_path.exists():
    msg = f'Data path "{args.data_path}" does not exist'
    raise FileNotFoundError(msg)
if args.data_path.is_file():
    msg = f'Data path "{args.data_path}" is a file, expected a directory'
    raise ValueError(msg)
if not args.data_path.joinpath(args.train_task, "train").exists():
    msg = f'Data path "{args.data_path.joinpath(args.train_task, "train")}"' + " does not exist"
    raise FileNotFoundError(msg)
if args.data_path.joinpath(args.train_task, "train").is_file():
    msg = f'Data path "{args.data_path.joinpath(args.train_task, "train")}"' + " is a file, expected a directory"
    raise ValueError(msg)
train_dataset = ONNXDataset(args.data_path.joinpath(args.train_task, "train"))
if args.train_size is not None and args.train_size < len(train_dataset):
    train_dataset = torch.utils.data.Subset(
        train_dataset,
        indices=np.random.RandomState(seed=args.seed).choice(len(train_dataset), size=args.train_size, replace=False),
    )

if not args.eval_path.exists():
    msg = f'Eval path "{args.eval_path}" does not exist'
    raise FileNotFoundError(msg)
if args.eval_path.is_file():
    msg = f'Eval path "{args.eval_path}" is a file, expected a directory'
    raise ValueError(msg)
if not args.eval_path.joinpath(args.eval_task, "val").exists():
    msg = f'Eval path "{args.eval_path.joinpath(args.eval_task, "val")}"' + " does not exist"
    raise FileNotFoundError(msg)
if args.eval_path.joinpath(args.eval_task, "val").is_file():
    msg = f'Eval path "{args.eval_path.joinpath(args.eval_task, "val")}"' + " is a file, expected a directory"
    raise ValueError(msg)
val_dataset = ONNXDataset(args.eval_path.joinpath(args.eval_task, "val"))

model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    num_labels=1,
    problem_type="regression",
    attn_implementation="flash_attention_2" if args.flash_attention else "sdpa",
).to(device)


def compute_metrics(eval_preds: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]) -> dict[str, float]:
    predictions, labels = eval_preds
    predictions = predictions.flatten()
    mae = mean_absolute_error(labels, predictions)
    spearman_corr = cast("float", stats.spearmanr(labels, predictions)[0])
    kendall_corr = cast("float", stats.kendalltau(labels, predictions)[0])
    return {"mae": mae, "spearman_corr": spearman_corr, "kendall_corr": kendall_corr}


# Train Setup
date = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d")
output_dir = args.output_path / date / run_name
output_dir.mkdir(exist_ok=True, parents=True)
with output_dir.joinpath("info.json").open("w", encoding="utf-8") as f:
    json.dump(vars(args), f, indent=4, default=str)

training_args = TrainingArguments(
    seed=args.seed,
    output_dir=output_dir,
    eval_strategy=args.eval_strategy,
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    weight_decay=args.weight_decay,
    warmup_ratio=args.warmup_ratio,
    logging_steps=200,
    lr_scheduler_type="polynomial",
    lr_scheduler_kwargs={"lr_end": args.lr * 0.1},
    gradient_accumulation_steps=1,
    push_to_hub=False,
    report_to="wandb",
    save_strategy=args.eval_strategy,
    run_name=run_name,
    load_best_model_at_end=True,
    save_steps=800 if args.eval_strategy == "steps" else None,  # pyright: ignore [reportArgumentType] # ty: ignore [invalid-argument-type]
    metric_for_best_model="kendall_corr",
    greater_is_better=True,
    bf16_full_eval=True,
    gradient_checkpointing=args.gradient_checkpointing,
)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
data_collator = DataCollatorWithPadding(tokenizer)
universal_trainer_params = {
    "model": model,
    "args": training_args,
    "train_dataset": train_dataset,
    "eval_dataset": val_dataset,
    # "processing_class": tokenizer,
    "compute_metrics": compute_metrics,
    "data_collator": data_collator,
}
if args.loss_fn == "mse":
    trainer = Trainer(**universal_trainer_params)
elif args.loss_fn == "huber":
    trainer = HuberTrainer(**universal_trainer_params, delta=5)
elif args.loss_fn == "mse_cap":
    trainer = MSECapTrainer(**universal_trainer_params, cap=10)
elif args.loss_fn == "pwr":
    trainer = PWRTrainer(**universal_trainer_params, compare_threshold=0.0, max_compare_ratio=4, margin=0.1)
# elif args.loss_fn == 'spearman':
#     trainer = SpearmanTrainer(**universal_trainer_params, tau=1.0)
elif args.loss_fn == "pwr_mining":
    trainer = PWRMiningTrainer(**universal_trainer_params, mining_mode="topk")  # ty: ignore [invalid-argument-type]
elif args.loss_fn == "pairlog":
    trainer = PairwiseLogTrainer(**universal_trainer_params)
elif args.loss_fn == "softmaxlist":
    trainer = SoftmaxListwiseTrainer(**universal_trainer_params)  # ty: ignore [invalid-argument-type]
elif args.loss_fn == "poly1list":
    trainer = Poly1ListwiseTrainer(**universal_trainer_params)  # ty: ignore [invalid-argument-type]
# elif args.loss_fn == 'plackett':
#     trainer = PlackettTrainer(**universal_trainer_params)
# elif args.loss_fn == 'approxrank':
#     trainer = ApproxRankTrainer(**universal_trainer_params, alpha=10.0)
else:
    msg = f"Loss function {args.loss_fn} not recognized"
    raise ValueError(msg)

trainer.train()

# Pred & Save
if args.save_pred:
    preds = trainer.predict(cast("torch.utils.data.Dataset[dict[str, Any]]", val_dataset))
    preds = pd.DataFrame(cast("npt.NDArray[np.float64]", preds.predictions).flatten(), columns=["pred"])  # ty: ignore [invalid-argument-type]
    preds["true"] = [x["accuracy"] for x in val_dataset]
    preds["dataset"] = [x["dataset"] for x in val_dataset]
    preds.to_csv(output_dir / "preds.csv", index=False)
