#!/usr/bin/env python3
"""
MethylReadBinProfiler
=====================

Extract read-level methylation features from bisulfite-sequencing BAM files in
fixed genomic bins.

The output format is intentionally kept compatible with the original research
script:

chrom  start  end  bin_mml:<methylated_C>:<unmethylated_C>:<bin_mean_methylation>
       average:<per-read methylation ratios>  mhl:<per-read MHL values>
       transitions_score:<per-read transition scores>  read_count

Notes
-----
- The BAM file is expected to contain Bismark-style XM tags, where uppercase
  "X" denotes methylated CpG cytosines and lowercase "x" denotes unmethylated
  CpG cytosines.
- To preserve compatibility with the original implementation, genomic bins are
  queried using the same coordinate convention as the legacy script.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
from pathlib import Path
from typing import Iterable, List, Sequence

import pysam
from pyfaidx import Fasta


DEFAULT_BIN_SIZE = 500
DEFAULT_CHROMOSOMES = [str(i) for i in range(1, 23)] + ["X", "Y"]
METHYLATION_PATTERN = re.compile(r"X|x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract read-level CpG methylation ratio, methylation haplotype "
            "load, and methylation transition score in fixed genomic bins."
        )
    )
    parser.add_argument(
        "-b",
        "--bam",
        required=True,
        help="Input coordinate-sorted and indexed BAM file.",
    )
    parser.add_argument(
        "-r",
        "--reference",
        required=True,
        help="Reference genome FASTA file used to define chromosome lengths.",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=DEFAULT_BIN_SIZE,
        help=f"Fixed genomic bin size. Default: {DEFAULT_BIN_SIZE}.",
    )
    parser.add_argument(
        "--chromosomes",
        nargs="+",
        default=DEFAULT_CHROMOSOMES,
        help=(
            "Chromosomes to process without the 'chr' prefix. "
            "Default: 1 2 ... 22 X Y."
        ),
    )
    parser.add_argument(
        "--tag",
        default="XM",
        help="BAM tag containing methylation calls. Default: XM.",
    )
    parser.add_argument(
        "--min-cpg",
        type=int,
        default=3,
        help="Minimum number of CpG calls required for read-level features. Default: 3.",
    )
    parser.add_argument(
        "--legacy-tag-index",
        type=int,
        default=None,
        help=(
            "Optional legacy mode: read the methylation string from read.tags[index] "
            "instead of using --tag. Use this only to reproduce older outputs exactly "
            "when BAM tag order is known."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level. Default: INFO.",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def extract_cpg_calls(methylation_string: str) -> List[str]:
    """Return CpG methylation calls encoded as X/x from a Bismark XM string."""
    return METHYLATION_PATTERN.findall(methylation_string)


def calculate_mhl(methylation_string: str) -> float:
    """
    Calculate methylation haplotype load using 1-, 2-, and 3-CpG blocks.

    This follows the formula used in the original script and is intended for
    reads with at least three CpG calls.
    """
    cpg_calls = extract_cpg_calls(methylation_string)
    if len(cpg_calls) < 3:
        raise ValueError("MHL requires at least three CpG calls.")

    base_1_total = len(cpg_calls)
    base_2_total = len(cpg_calls) - 1
    base_3_total = len(cpg_calls) - 2

    base_1_methylated = sum(1 for base in cpg_calls if base == "X")
    base_2_methylated = sum(
        1 for idx in range(base_2_total) if cpg_calls[idx : idx + 2] == ["X", "X"]
    )
    base_3_methylated = sum(
        1 for idx in range(base_3_total) if cpg_calls[idx : idx + 3] == ["X", "X", "X"]
    )

    return (
        1 * base_1_methylated / base_1_total
        + 2 * base_2_methylated / base_2_total
        + 3 * base_3_methylated / base_3_total
    ) / (1 + 2 + 3)


def calculate_transition_score_ratio(methylation_string: str) -> float:
    """
    Calculate the fraction of adjacent CpG calls that switch methylation state.

    For example, XxXX has transitions at X->x and x->X, resulting in a score of
    2 / 3.
    """
    cpg_calls = extract_cpg_calls(methylation_string)
    if len(cpg_calls) < 2:
        raise ValueError("Transition score requires at least two CpG calls.")

    transition_count = sum(
        1 for idx in range(1, len(cpg_calls)) if cpg_calls[idx] != cpg_calls[idx - 1]
    )
    return transition_count / (len(cpg_calls) - 1)


def get_methylation_string(
    read: pysam.AlignedSegment,
    tag: str = "XM",
    legacy_tag_index: int | None = None,
) -> str:
    """Extract the methylation string from a BAM read."""
    if legacy_tag_index is not None:
        return read.tags[legacy_tag_index][1]
    return read.get_tag(tag)


def get_chromosome_length(reference: Fasta, chrom_name: str) -> int:
    """
    Return chromosome length using the legacy pyfaidx convention.

    The expression reference[chrom][-1].start is retained to match the original
    implementation as closely as possible.
    """
    return int(reference[chrom_name][-1].start)


def iter_chromosome_names(chromosomes: Sequence[str]) -> Iterable[str]:
    for chrom in chromosomes:
        chrom = str(chrom)
        yield chrom if chrom.startswith("chr") else f"chr{chrom}"


def build_output_path(bam_path: Path, outdir: Path, bin_size: int) -> Path:
    return outdir / f"{bam_path.stem}_{bin_size}_aver_mhl_ts_3cpg.txt"


def process_bam(
    bam_path: Path,
    reference_path: Path,
    outdir: Path,
    bin_size: int = DEFAULT_BIN_SIZE,
    chromosomes: Sequence[str] = DEFAULT_CHROMOSOMES,
    tag: str = "XM",
    min_cpg: int = 3,
    legacy_tag_index: int | None = None,
) -> Path:
    """Process one BAM file and return the generated output path."""
    outdir.mkdir(parents=True, exist_ok=True)
    output_path = build_output_path(bam_path, outdir, bin_size)

    logging.info("Loading reference: %s", reference_path)
    reference = Fasta(str(reference_path))

    logging.info("Processing BAM: %s", bam_path)
    logging.info("Writing output: %s", output_path)

    with pysam.AlignmentFile(str(bam_path), "rb") as bam_file, output_path.open("w") as output:
        for chrom_name in iter_chromosome_names(chromosomes):
            try:
                chrom_length = get_chromosome_length(reference, chrom_name)
            except KeyError:
                logging.warning("Chromosome %s not found in reference; skipped.", chrom_name)
                continue

            region_count = int(math.ceil(chrom_length / bin_size))
            logging.info("%s: %s bins", chrom_name, region_count)

            for bin_index in range(region_count):
                bin_start = bin_index * bin_size + 1
                bin_end = (bin_index + 1) * bin_size

                average_ratio = "average"
                mhl_ratio = "mhl"
                transition_score = "transitions_score"
                read_count = 0
                bin_unmethylated_c = 0
                bin_methylated_c = 0

                # The fetch coordinates are intentionally kept consistent with
                # the original script to avoid changing existing outputs.
                for read in bam_file.fetch(chrom_name, bin_start, bin_end):
                    try:
                        methylation_string = get_methylation_string(
                            read, tag=tag, legacy_tag_index=legacy_tag_index
                        )
                    except (KeyError, IndexError):
                        continue

                    unmethylated_c = methylation_string.count("x")
                    methylated_c = methylation_string.count("X")
                    total_cpg = methylated_c + unmethylated_c

                    bin_unmethylated_c += unmethylated_c
                    bin_methylated_c += methylated_c

                    if total_cpg >= min_cpg:
                        per_read_ratio = methylated_c / total_cpg
                        average_ratio += f":{per_read_ratio:.3f}"
                        read_count += 1

                        mhl_value = calculate_mhl(methylation_string)
                        mhl_ratio += f":{mhl_value:.3f}"

                        transition_value = calculate_transition_score_ratio(methylation_string)
                        transition_score += f":{transition_value:.3f}"

                if bin_methylated_c + bin_unmethylated_c == 0:
                    bin_mean_methylation = "NA"
                else:
                    bin_mean_methylation = bin_methylated_c / (
                        bin_methylated_c + bin_unmethylated_c
                    )

                output.write(
                    f"{chrom_name}\t{bin_start}\t{bin_end}\t"
                    f"bin_mml:{bin_methylated_c}:{bin_unmethylated_c}:{bin_mean_methylation}\t"
                    f"{average_ratio}\t{mhl_ratio}\t{transition_score}\t{read_count}\n"
                )

    return output_path


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    bam_path = Path(args.bam)
    reference_path = Path(args.reference)
    outdir = Path(args.outdir)

    if not bam_path.exists():
        raise FileNotFoundError(f"Input BAM not found: {bam_path}")
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference FASTA not found: {reference_path}")

    output_path = process_bam(
        bam_path=bam_path,
        reference_path=reference_path,
        outdir=outdir,
        bin_size=args.bin_size,
        chromosomes=args.chromosomes,
        tag=args.tag,
        min_cpg=args.min_cpg,
        legacy_tag_index=args.legacy_tag_index,
    )
    logging.info("Done: %s", output_path)


if __name__ == "__main__":
    main()
