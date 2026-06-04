#!/usr/bin/env python3
"""Train a CNN-BiLSTM model on read-level methylation encodings.

This script trains the read-level classifier used downstream for methylation
signal detection. It preserves the core architecture, class-balancing strategy,
train/test split, output file names, and default hyperparameters of the original
research script, while replacing local paths and positional arguments with a
public, reproducible command-line interface.
"""

from __future__ import annotations

import argparse
import gc
import logging
import random
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

LOGGER = logging.getLogger("train_read_methylation_cnn_bilstm")

EncodedReadData = Tuple[
    List[str], List[str], List[str], List[str],
    List[str], List[str], List[str], List[str],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a CNN-BiLSTM classifier using fixed-length read sequences "
            "and binary methylation encodings."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help=(
            "Input directory containing ref_switch_reads/ and tumor_switch_reads/ "
            "subdirectories with *_reads_clean_reads.txt files."
        ),
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help=(
            "Output directory for model weights and prediction files. "
            "Default: <input-dir>/train_out."
        ),
    )
    parser.add_argument("--read-length", type=int, default=140, help="Expected read length. Default: 140.")
    parser.add_argument("--min-cpg", type=int, default=3, help="Minimum CpG count required per read. Default: 3.")
    parser.add_argument("--epochs", type=int, default=150, help="Maximum number of training epochs. Default: 150.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size. Default: 128.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam learning rate. Default: 1e-4.")
    parser.add_argument("--validation-split", type=float, default=0.1, help="Validation split within training data. Default: 0.1.")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Fraction of balanced reads used for training. Default: 0.8.")
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience. Default: 10.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed. If omitted, stochastic behavior follows the original script.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress messages.")
    return parser.parse_args()


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="[%(levelname)s] %(message)s",
    )


def set_random_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def count_cpg(sequence: str) -> int:
    return sum(1 for i in range(len(sequence) - 1) if sequence[i] == "C" and sequence[i + 1] == "G")


def iter_clean_read_files(input_dir: Path) -> Tuple[List[Path], List[Path]]:
    ref_files = sorted((input_dir / "ref_switch_reads").glob("*_reads_clean_reads.txt"))
    tumor_files = sorted((input_dir / "tumor_switch_reads").glob("*_reads_clean_reads.txt"))
    if not ref_files:
        raise FileNotFoundError(f"No reference read files found under {input_dir / 'ref_switch_reads'}")
    if not tumor_files:
        raise FileNotFoundError(f"No tumor read files found under {input_dir / 'tumor_switch_reads'}")
    return ref_files, tumor_files


def load_group_reads(files: Iterable[Path], read_length: int, min_cpg: int) -> Tuple[List[str], List[str], List[str], List[str]]:
    sequences: List[str] = []
    methylation: List[str] = []
    chromosomes: List[str] = []
    regions: List[str] = []

    for file_path in files:
        LOGGER.info("Reading %s", file_path)
        with file_path.open("r") as handle:
            for line in handle:
                fields = line.strip().split()
                if len(fields) < 4:
                    continue
                chrom, region, seq, methy = fields[:4]
                if len(seq) != read_length or len(methy) != read_length:
                    continue
                if count_cpg(seq) < min_cpg:
                    continue
                chromosomes.append(chrom)
                regions.append(region)
                sequences.append(seq)
                methylation.append(methy)

    return sequences, methylation, chromosomes, regions


def prepare_read_data(input_dir: Path, read_length: int, min_cpg: int) -> EncodedReadData:
    ref_files, tumor_files = iter_clean_read_files(input_dir)
    normal_seq, normal_methy, normal_chrom, normal_region = load_group_reads(ref_files, read_length, min_cpg)
    tumor_seq, tumor_methy, tumor_chrom, tumor_region = load_group_reads(tumor_files, read_length, min_cpg)

    if not normal_seq:
        raise ValueError("No eligible reference reads remained after filtering.")
    if not tumor_seq:
        raise ValueError("No eligible tumor reads remained after filtering.")

    LOGGER.info("Loaded %d reference reads and %d tumor reads", len(normal_seq), len(tumor_seq))
    return normal_seq, normal_methy, normal_chrom, normal_region, tumor_seq, tumor_methy, tumor_chrom, tumor_region


def sequence_to_integer_matrix(sequences: Sequence[str], methylation: Sequence[str], read_length: int) -> np.ndarray:
    """Encode A/T/C/G as 0/1/2/3 and methylated CpG cytosines as 4."""
    encoded = np.zeros((len(sequences), read_length), dtype="int")
    base_map = {"A": 0, "T": 1, "C": 2, "G": 3}

    for i, seq in enumerate(sequences):
        for j, base in enumerate(seq[:read_length]):
            encoded[i, j] = base_map.get(base, 3)
            if int(methylation[i][j]) == 1:
                encoded[i, j] = 4
    return encoded


def integer_matrix_to_onehot(encoded: np.ndarray, read_length: int) -> np.ndarray:
    """Convert integer encoding to a 5-channel one-hot methylation representation."""
    module = np.array(
        [
            [1, 0, 0, 0, 0],  # A
            [0, 1, 0, 0, 0],  # T
            [0, 0, 1, 0, 0],  # C
            [0, 0, 0, 1, 0],  # G
            [0, 0, 1, 0, 1],  # methylated C
        ],
        dtype="int",
    )
    onehot = np.zeros((len(encoded), read_length, 5), dtype="int")
    for i, read in enumerate(encoded):
        onehot[i] = module[read]
    return onehot


def make_chrom_region(chromosomes: Sequence[str], regions: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    chrom = np.zeros(len(chromosomes), dtype="int")
    region = np.zeros(len(regions), dtype="int")

    for i, chrom_name in enumerate(chromosomes):
        match = re.findall(r"(\d+)", chrom_name)
        if not match:
            raise ValueError(f"Unable to parse chromosome number from: {chrom_name}")
        chrom[i] = int(match[0])
        region[i] = int(regions[i])
    return chrom, region


def build_cnn_bilstm_model(read_length: int = 140) -> Sequential:
    """Build the CNN-BiLSTM architecture used for read-level classification."""
    model = Sequential(name="ReadMethylationCNNBiLSTM")
    model.add(
        layers.Conv1D(
            filters=100,
            kernel_size=10,
            padding="same",
            activation="relu",
            strides=1,
            input_shape=(read_length, 5),
        )
    )
    model.add(layers.MaxPooling1D(pool_size=2, strides=2))
    model.add(layers.Dropout(0.2))
    model.add(layers.Bidirectional(layers.LSTM(33, return_sequences=True)))
    model.add(layers.Conv1D(filters=100, kernel_size=3, padding="same", activation="relu", strides=1))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(750, activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(300, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


def enable_gpu_memory_growth() -> None:
    for gpu in tf.config.experimental.list_physical_devices(device_type="GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            LOGGER.warning("Could not set memory growth for %s: %s", gpu, exc)


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    set_random_seed(args.seed)

    input_dir = Path(args.input_dir).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else input_dir / "train_out"
    outdir.mkdir(parents=True, exist_ok=True)

    (
        normal_seq,
        normal_methy,
        normal_chrom_raw,
        normal_region_raw,
        tumor_seq,
        tumor_methy,
        tumor_chrom_raw,
        tumor_region_raw,
    ) = prepare_read_data(input_dir, args.read_length, args.min_cpg)

    balanced_n = min(len(tumor_chrom_raw), len(normal_chrom_raw))
    LOGGER.info("Using %d reads per class after class balancing", balanced_n)

    normal_seq_encoded = sequence_to_integer_matrix(normal_seq, normal_methy, args.read_length)
    tumor_seq_encoded = sequence_to_integer_matrix(tumor_seq, tumor_methy, args.read_length)
    normal_chrom, normal_region = make_chrom_region(normal_chrom_raw, normal_region_raw)
    tumor_chrom, tumor_region = make_chrom_region(tumor_chrom_raw, tumor_region_raw)

    normal_perm = random.sample(range(len(normal_seq)), len(normal_seq))
    normal_seq_encoded = normal_seq_encoded[normal_perm]
    normal_chrom = normal_chrom[normal_perm]
    normal_region = normal_region[normal_perm]

    normal_onehot = integer_matrix_to_onehot(normal_seq_encoded[:balanced_n], args.read_length)
    tumor_onehot = integer_matrix_to_onehot(tumor_seq_encoded[:balanced_n], args.read_length)

    total_balanced = len(normal_seq[:balanced_n]) + len(tumor_seq[:balanced_n])
    mixed_perm = random.sample(range(total_balanced), total_balanced)

    data_lstm_all = np.vstack((normal_seq_encoded[:balanced_n], tumor_seq_encoded[:balanced_n]))
    data = np.vstack((normal_onehot[:balanced_n], tumor_onehot[:balanced_n]))
    label_all = np.array([0] * balanced_n + [1] * balanced_n)
    chrom_all = np.hstack((normal_chrom[:balanced_n], tumor_chrom[:balanced_n]))
    region_all = np.hstack((normal_region[:balanced_n], tumor_region[:balanced_n]))

    chrom_all = chrom_all[mixed_perm]
    region_all = region_all[mixed_perm]
    data = data[mixed_perm]
    data_lstm_all = data_lstm_all[mixed_perm]
    label_all = label_all[mixed_perm]

    del normal_seq_encoded, tumor_seq_encoded, normal_onehot, tumor_onehot
    del normal_seq, normal_methy, normal_chrom_raw, normal_region_raw
    del tumor_seq, tumor_methy, tumor_chrom_raw, tumor_region_raw, normal_perm
    gc.collect()

    train_num = int(len(data) * args.train_fraction)
    test_num = int(len(data) * (1 - args.train_fraction))

    train_data = data[:train_num]
    train_label = label_all[:train_num]
    test_data = data[train_num:(train_num + test_num)]
    test_label = label_all[train_num:(train_num + test_num)]
    test_chrom = chrom_all[train_num:(train_num + test_num)]
    test_region = region_all[train_num:(train_num + test_num)]

    data_lstm = data_lstm_all[:(train_num + test_num)]
    label = label_all[:(train_num + test_num)]
    chrom = chrom_all[:(train_num + test_num)]
    region = region_all[:(train_num + test_num)]
    data_origin = data_lstm_all[(train_num + test_num):]
    label_origin = label_all[(train_num + test_num):]
    chrom_origin = chrom_all[(train_num + test_num):]
    region_origin = region_all[(train_num + test_num):]

    np.savetxt(outdir / "data_origin_lstm.txt", data_origin)
    np.savetxt(outdir / "label_origin.txt", label_origin)
    np.savetxt(outdir / "chrom_origin.txt", chrom_origin)
    np.savetxt(outdir / "region_origin.txt", region_origin)

    enable_gpu_memory_growth()
    model = build_cnn_bilstm_model(read_length=args.read_length)
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    weight_path = outdir / "weight.h5"
    history = model.fit(
        train_data,
        train_label,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=args.patience),
            ModelCheckpoint(filepath=str(weight_path), save_best_only=True),
        ],
        shuffle=True,
        verbose=2 if args.verbose else 1,
    )
    del history

    model.load_weights(weight_path)
    result = model.predict(test_data, verbose=0)
    np.savetxt(outdir / "test_label.txt", test_label)
    np.savetxt(outdir / "predict_result.txt", result)
    np.savetxt(outdir / "test_chrom.txt", test_chrom)
    np.savetxt(outdir / "test_region.txt", test_region)

    result_all = model.predict(data[:(train_num + test_num)], verbose=0)
    np.savetxt(outdir / "data_lstm.txt", data_lstm)
    np.savetxt(outdir / "label_all.txt", label)
    np.savetxt(outdir / "chrom_all.txt", chrom)
    np.savetxt(outdir / "region_all.txt", region)
    np.savetxt(outdir / "result_all.txt", result_all)

    LOGGER.info("Training complete. Outputs written to %s", outdir)


if __name__ == "__main__":
    main()
