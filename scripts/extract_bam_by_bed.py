#!/usr/bin/env python3
"""Extract target genomic regions from multiple BAM files using a BED file.

This script is a GitHub-ready, de-identified version of an internal utility used
for extracting reads from BAM files overlapping predefined genomic intervals.
It preserves the original output naming pattern by default:

    <input_basename>.switch.bam

For each output BAM file, a BAM index is generated using ``samtools index``.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-extract reads from BAM files overlapping a BED file and "
            "create indexed region-specific BAM files."
        )
    )
    parser.add_argument(
        "--bed",
        required=True,
        help="BED file containing target genomic intervals.",
    )
    parser.add_argument(
        "--bam-dir",
        required=True,
        help="Directory containing input BAM files.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Directory for output BAM and BAI files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.bam",
        help="Input BAM filename pattern. Default: *.bam",
    )
    parser.add_argument(
        "--suffix",
        default=".switch.bam",
        help="Suffix appended to the input BAM basename. Default: .switch.bam",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=min(8, os.cpu_count() or 8),
        help="Number of parallel BAM files to process. Default: min(8, CPU count).",
    )
    parser.add_argument(
        "--samtools",
        default="samtools",
        help="Path to the samtools executable. Default: samtools",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output BAM files. By default, existing outputs are skipped.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress messages.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def discover_bam_files(bam_dir: Path, pattern: str) -> List[Path]:
    bam_files = sorted(bam_dir.glob(pattern))
    return [path for path in bam_files if path.is_file()]


def build_output_path(input_bam: Path, outdir: Path, suffix: str) -> Path:
    if input_bam.name.endswith(".bam"):
        sample_name = input_bam.name[: -len(".bam")]
    else:
        sample_name = input_bam.stem
    return outdir / f"{sample_name}{suffix}"


def run_command(command: Sequence[str], *, stdout_path: Path | None = None) -> None:
    logging.debug("Running command: %s", " ".join(command))
    if stdout_path is None:
        subprocess.run(command, check=True)
    else:
        with stdout_path.open("wb") as stdout_handle:
            subprocess.run(command, stdout=stdout_handle, check=True)


def process_one_bam(task: Tuple[str, str, str, str, str, bool]) -> Tuple[str, str]:
    bam_file_str, bed_file_str, outdir_str, suffix, samtools, overwrite = task
    bam_file = Path(bam_file_str)
    bed_file = Path(bed_file_str)
    outdir = Path(outdir_str)
    output_bam = build_output_path(bam_file, outdir, suffix)

    if output_bam.exists() and not overwrite:
        return str(bam_file), "skipped_existing"

    view_command = [
        samtools,
        "view",
        "-b",
        "-h",
        str(bam_file),
        "-L",
        str(bed_file),
    ]
    index_command = [samtools, "index", str(output_bam)]

    try:
        run_command(view_command, stdout_path=output_bam)
        run_command(index_command)
        return str(bam_file), "done"
    except subprocess.CalledProcessError as exc:
        if output_bam.exists():
            output_bam.unlink()
        return str(bam_file), f"failed: {exc}"


def validate_inputs(bed_file: Path, bam_dir: Path, outdir: Path, processes: int) -> None:
    if not bed_file.exists():
        raise FileNotFoundError(f"BED file not found: {bed_file}")
    if not bam_dir.exists():
        raise FileNotFoundError(f"BAM directory not found: {bam_dir}")
    if not bam_dir.is_dir():
        raise NotADirectoryError(f"BAM input path is not a directory: {bam_dir}")
    if processes < 1:
        raise ValueError("--processes must be at least 1")
    outdir.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    bed_file = Path(args.bed).resolve()
    bam_dir = Path(args.bam_dir).resolve()
    outdir = Path(args.outdir).resolve()

    try:
        validate_inputs(bed_file, bam_dir, outdir, args.processes)
    except Exception as exc:  # noqa: BLE001 - provide user-friendly CLI error
        logging.error(str(exc))
        return 1

    bam_files = discover_bam_files(bam_dir, args.pattern)
    if not bam_files:
        logging.error("No BAM files found in %s using pattern %r", bam_dir, args.pattern)
        return 1

    logging.info("Found %d BAM file(s).", len(bam_files))
    logging.info("Using %d parallel process(es).", args.processes)

    tasks = [
        (
            str(bam_file),
            str(bed_file),
            str(outdir),
            args.suffix,
            args.samtools,
            args.overwrite,
        )
        for bam_file in bam_files
    ]

    with Pool(args.processes) as pool:
        results = pool.map(process_one_bam, tasks)

    failed = 0
    for bam_file, status in results:
        if status.startswith("failed"):
            failed += 1
            logging.error("%s: %s", bam_file, status)
        else:
            logging.info("%s: %s", bam_file, status)

    if failed:
        logging.error("Completed with %d failed BAM file(s).", failed)
        return 1

    logging.info("All BAM files were processed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
