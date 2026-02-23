import argparse
import logging
import multiprocessing
from pathlib import Path
import sys
import warnings

import numpy as np
import onnx
import onnxsim
import tqdm
import tqdm.contrib.logging

from onnxnet.lmdb_dataset import LMDBDataset, LMDBDatasetWriter
from onnxnet.process.onnx_graph_utils import encode_graph, onnx_to_graph


def _process_onnx_file(args: tuple[argparse.Namespace, Path]) -> None:
    """Preprocess ONNX models for encoding."""
    logger = logging.getLogger(__name__)
    try:
        onnx_model = onnx.load(args[1])
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        onnx_model = onnxsim.simplify(onnx_model)[0]
        graph = onnx_to_graph(onnx_model, ignore_multiple_inputs=True)
        pos_enc = graph.generate_position_encodings(k_frac=args[0].k_frac)
        model_str, char_node_ids = encode_graph(graph, return_node_ids=True)
    except Exception as e:  # noqa: BLE001
        logger = logging.getLogger(__name__)
        logger.error('Error processing file "%s": %s', args[1], e, exc_info=e)  # noqa: TRY400
        return
    try:
        acc = float(onnx_model.metadata_props[0].value)
        with LMDBDatasetWriter(args[0].out_file) as writer:
            writer.add({
                "file": args[1].relative_to(args[0].in_dir).with_suffix("").as_posix(),
                "accuracy": acc,
                "dataset": "cifar10",
                "model_str": model_str,
                "char_indices": np.array(char_node_ids),
                "node_pos_encs": pos_enc,
            })
    except Exception as e:  # noqa: BLE001
        logger = logging.getLogger(__name__)
        logger.error('Error saving preprocessing results for file "%s": %s', args[1], e, exc_info=e)  # noqa: TRY400


def main() -> None:  # noqa: C901
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
        "-o",
        "--out-file",
        type=Path,
        default=None,
        help="File to save the preprocessed data in.",
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
        "-k",
        dest="k_frac",
        type=float,
        default=0.2,
        help="Latent code size as percentage of graph node count.",
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
    if args.out_file is None:
        args.out_file = args.in_dir.parent.joinpath(f"{args.in_dir.name}_preprocessed").with_suffix(".lmdb")
        logger.info('Output file not specified, using "%s"', args.out_file)
    existing_outputs = set()
    if args.out_file.exists():
        if not args.out_file.is_file():
            logger.critical('Output path "%s" exists and is not a file', args.out_file)
            sys.exit(-1)
        elif args.existing_output_policy == "error":
            logger.critical('Output path "%s" already exists', args.out_file)
            sys.exit(-1)
        elif args.existing_output_policy == "delete":
            logger.info('Removing output file "%s"', args.out_file)
            args.out_file.unlink()
        elif args.existing_output_policy == "skip":
            logger.info("Collecting existing output files")
            dataset = LMDBDataset(args.in_file)
            for sample in dataset:  # noqa: FURB142
                existing_outputs.add(sample["file"])

    todo = []
    for in_file in args.in_dir.glob("**/*.onnx"):
        if in_file.relative_to(args.in_dir).with_suffix("").as_posix() in existing_outputs:
            continue
        todo.append(in_file)
    logger.info("Found %d files to process", len(todo))

    logger.info("Preprocessing files")
    process_args = [(args, file) for file in todo]
    with (
        multiprocessing.Pool(args.n_procs) as process_pool,
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
