import argparse
import csv
import logging
import multiprocessing
import multiprocessing.synchronize
from pathlib import Path
import pickle  # noqa: S403
import shutil
import sys
import warnings

import numpy as np
import numpy.typing as npt
import onnx
import onnxsim
from pydot import cast
import tqdm
import tqdm.contrib.logging
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from onnxnet.process.onnx_graph_utils import encode_graph, onnx_to_graph


def _init_process(out_file_lock: multiprocessing.synchronize.Lock) -> None:
    global lock  # noqa: PLW0603  # ty: ignore [unresolved-global]
    lock = out_file_lock


def _process_onnx_file(args: tuple[argparse.Namespace, Path, PreTrainedTokenizerBase, npt.NDArray[np.float64]]) -> None:
    """Preprocess ONNX models for encoding."""
    logger = logging.getLogger(__name__)
    try:
        onnx_model = onnx.load(args[1])
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        onnx_model = onnxsim.simplify(onnx_model)[0]
        graph = onnx_to_graph(onnx_model, ignore_multiple_inputs=True)
        pos_enc = graph.generate_position_encodings(k_frac=0.2)
        model_str, char_node_ids = encode_graph(graph, return_node_ids=True)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error('Error processing file "%s": %s', args[1], e, exc_info=e)  # noqa: TRY400
        return
    try:
        acc = float(onnx_model.metadata_props[0].value)
        tknzer_outp = args[2](model_str, truncation=True, return_offsets_mapping=True)
        with (
            lock,  # ty: ignore [unresolved-reference]
            args[0].out_dir.joinpath("metadata.csv").open("a", encoding="UTF-8", newline="") as f,
        ):
            csvwriter = csv.writer(f)
            if f.tell() == 0:
                csvwriter.writerow(["file_path", "accuracy", "num_tokens", "dataset"])
            csvwriter.writerow([
                args[1].relative_to(args[0].in_dir).with_suffix("").as_posix(),
                acc,
                len(cast("list[int]", tknzer_outp["input_ids"])),
                "cifar10",
            ])
        base_file_name = args[1].relative_to(args[0].in_dir).with_suffix("").as_posix()
        with args[0].out_dir.joinpath(base_file_name).with_suffix(".ndid").open("wb") as f:
            token_node_ids = []
            for start, end in cast("list[tuple[int, int]]", tknzer_outp["offset_mapping"]):
                if start == end == 0:
                    token_node_ids.append(None)
                else:
                    token_node_ids.append(char_node_ids[start])
            pickle.dump(token_node_ids, f)
        del tknzer_outp["offset_mapping"]
        with args[0].out_dir.joinpath(base_file_name).with_suffix(".tkid").open("wb") as f:
            pickle.dump(tknzer_outp, f)
        pos_enc = np.pad(pos_enc[:, :256], [(0, 0), (0, max(0, 256 - pos_enc.shape[1]))])
        pos_enc = args[3] @ pos_enc[..., None]
        np.save(args[0].out_dir.joinpath(base_file_name).with_suffix(""), pos_enc[..., 0])
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error('Error saving preprocessing results for file "%s": %s', args[1], e, exc_info=e)  # noqa: TRY400


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Preprocess ONNX models for encoding."""
    argparser = argparse.ArgumentParser(description="Preprocess ONNX models for encoding.")
    _ = argparser.add_argument(
        "-f",
        "--format",
        choices=["v2"],
        default="v2",
        help="Format of the preprocessed data.",
    )
    _ = argparser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        default="Qwen/Qwen3-0.6B",  # "answerdotai/ModernBERT-base",
        help="Pretrained tokenizer to use.",
    )
    _ = argparser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to save the preprocessed data in.",
    )
    _ = argparser.add_argument(
        "-p",
        "--existing-output-policy",
        choices=["delete", "skip", "error"],
        default="error",
        help="Policy for handling existing output files.",
    )
    _ = argparser.add_argument(
        "-n",
        "--n-procs",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of parallel processes to use.",
    )
    _ = argparser.add_argument(
        "in_dir",
        metavar="in-dir",
        type=Path,
        help="Directory containing ONNX models to preprocess.",
    )
    args = argparser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname).4s]: %(message)s",
        datefmt="%Y.%m.%d %H:%M:%S",
    )
    warnings.showwarning = lambda m, _, __, ___, ____, _____: logging.getLogger("py.warnings").warning(m)  # ty: ignore [invalid-assignment]
    logger = logging.getLogger(__name__)

    logger.info("Validating CLI arguments")
    if not args.in_dir.exists():
        logger.critical('Input directory "%s" does not exist', args.in_dir)
        sys.exit(-1)
    elif not args.in_dir.is_dir():
        logger.critical('Input directory "%s" is not a directory', args.in_dir)
        sys.exit(-1)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    except OSError as e:
        logger.critical('Failed to load tokenizer "%s"', args.tokenizer, exc_info=e)
        sys.exit(-1)
    if args.out_dir is None:
        args.out_dir = args.in_dir.parent / f"{args.in_dir.name}_preprocessed"
        logger.info('Output directory not specified, using "%s"', args.out_dir)
    existing_outputs = set()
    if args.out_dir.exists():
        if not args.out_dir.is_dir():
            logger.critical('Output directory "%s" exists and is not a directory', args.out_dir)
            sys.exit(-1)
        elif args.existing_output_policy == "error":
            logger.critical('Output directory "%s" already exists', args.out_dir)
            sys.exit(-1)
        elif args.existing_output_policy == "delete":
            logger.info('Clearing output directory "%s"', args.out_dir)
            for out_file in args.out_dir.glob("**/*"):
                if out_file.is_file():
                    out_file.unlink()
                elif out_file.is_dir():
                    shutil.rmtree(out_file)
        elif args.existing_output_policy == "skip":
            logger.info("Collecting existing output files")
            for out_file in args.out_dir.glob("**/*.npy"):  # noqa: FURB142
                existing_outputs.add(out_file.relative_to(args.out_dir).with_suffix("").as_posix())
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    for in_file in args.in_dir.glob("**/*.onnx"):
        if in_file.relative_to(args.in_dir).with_suffix("").as_posix() in existing_outputs:
            continue
        todo.append(in_file)
    logger.info("Found %d files to process", len(todo))

    logger.info("Preprocessing files")
    out_file_lock = multiprocessing.Lock()
    random_projection = np.random.default_rng().standard_normal((512, 256)) / np.sqrt(256)
    process_args = [(args, file, tokenizer, random_projection) for file in todo]
    with (
        multiprocessing.Pool(args.n_procs, initializer=_init_process, initargs=(out_file_lock,)) as process_pool,
        tqdm.contrib.logging.logging_redirect_tqdm(),
    ):
        _ = list(
            tqdm.auto.tqdm(  # ty: ignore [possibly-missing-attribute]
                process_pool.imap_unordered(
                    _process_onnx_file,
                    process_args,
                ),
                total=len(process_args),
                desc="Preprocessing ONNX models",
                unit="file",
            ),
        )
    logger.info("Done")


if __name__ == "__main__":
    main()
