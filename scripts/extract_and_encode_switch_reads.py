#!/usr/bin/env python3
"""
Extract DMR-overlapping reads from BAM files and encode fixed-length methylation patterns.

This script supports a common workflow used in read-level methylation modeling:
1. Rank hypo- and hypermethylated DMRs using abnormal-read ratios.
2. Optionally export top-ranked DMR sets as BED files.
3. Extract reads overlapping each BED set from one or more BAM files.
4. Convert read methylation strings into fixed-length binary methylation encodings.

The default settings preserve the original analysis behavior as closely as possible while
removing local paths and making all directories/parameters explicit.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import pysam


DEFAULT_DMR_COLUMNS = [
    "keys",
    "min_diff",
    "max_diff",
    "mean_diff",
    "ref_max",
    "ref_min",
    "region",
    "total_sum",
    "above_sum",
    "below_sum",
    "above_ratio",
    "below_ratio",
]

DEFAULT_DMR_SETS = (
    "hypo_100,hypo_200,hypo_500,hypo_1000,hypo_2000,hypo_3000,hypo_4000,hypo_5000,"
    "hyper_100,hyper_200,hyper_500,hyper_1000,hyper_1500,hyper_2000,hyper_3000"
)


@dataclass(frozen=True)
class DmrSet:
    direction: str
    top_n: int

    @property
    def name(self) -> str:
        return f"{self.direction}_{self.top_n}"


def parse_dmr_sets(raw_sets: str) -> list[DmrSet]:
    """Parse comma-separated DMR set labels such as 'hypo_100,hyper_500'."""
    parsed: list[DmrSet] = []
    for item in raw_sets.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            direction, number = item.split("_", 1)
            if direction not in {"hypo", "hyper"}:
                raise ValueError
            parsed.append(DmrSet(direction=direction, top_n=int(number)))
        except ValueError as exc:
            raise ValueError(
                f"Invalid DMR set '{item}'. Expected format like 'hypo_100' or 'hyper_500'."
            ) from exc
    return parsed


def load_dmr_table(dmr_file: Path) -> pd.DataFrame:
    """Load and validate a DMR statistics table."""
    dmr_df = pd.read_csv(dmr_file, sep="\t")
    missing = [col for col in DEFAULT_DMR_COLUMNS if col not in dmr_df.columns]
    if missing:
        raise ValueError(f"DMR file is missing required columns: {', '.join(missing)}")
    return dmr_df[DEFAULT_DMR_COLUMNS].copy()


def split_keys_to_bed(keys: pd.Series) -> pd.DataFrame:
    """Convert keys formatted as chr_start_end into a BED-like DataFrame."""
    bed_df = keys.astype(str).str.split("_", expand=True)
    if bed_df.shape[1] != 3:
        raise ValueError("The 'keys' column must use the format chr_start_end.")
    bed_df.columns = ["chr", "start", "end"]
    return bed_df


def build_ranked_dmr_tables(dmr_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create ranked hypo- and hyper-DMR BED tables."""
    hypo_df = dmr_df[dmr_df["region"].astype(str).str.contains("hypo", na=False)].copy()
    hypo_df.sort_values("below_ratio", ascending=False, inplace=True)

    hyper_df = dmr_df[dmr_df["region"].astype(str).str.contains("hyper", na=False)].copy()
    hyper_df.sort_values("above_ratio", ascending=False, inplace=True)

    return {
        "hypo": split_keys_to_bed(hypo_df["keys"]),
        "hyper": split_keys_to_bed(hyper_df["keys"]),
    }


def export_top_dmr_beds(
    ranked_beds: dict[str, pd.DataFrame],
    dmr_sets: Iterable[DmrSet],
    outdir: Path,
    overwrite: bool = False,
) -> list[Path]:
    """Export top-ranked DMR sets as BED files."""
    bed_paths: list[Path] = []
    for dmr_set in dmr_sets:
        set_dir = outdir / dmr_set.name
        set_dir.mkdir(parents=True, exist_ok=True)
        bed_path = set_dir / f"{dmr_set.name}.bed"
        if bed_path.exists() and not overwrite:
            logging.info("Using existing BED file: %s", bed_path)
        else:
            ranked_beds[dmr_set.direction].head(dmr_set.top_n).to_csv(
                bed_path, sep="\t", header=False, index=False
            )
            logging.info("Wrote BED file: %s", bed_path)
        bed_paths.append(bed_path)
    return bed_paths


def find_bam_files(bam_dir: Path, pattern: str = "*.bam") -> list[Path]:
    """Return sorted BAM files from an input directory."""
    bam_files = sorted(bam_dir.glob(pattern))
    if not bam_files:
        raise FileNotFoundError(f"No BAM files matching '{pattern}' were found in {bam_dir}")
    return bam_files


def build_dmr_bounds(dmr_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Create a dictionary from DMR key to reference min/max bounds."""
    bounds = {}
    for row in dmr_df.itertuples(index=False):
        bounds[str(row.keys)] = (float(row.ref_min), float(row.ref_max))
    return bounds


def parse_read_sequence_from_xg(xg_tag: str) -> Optional[tuple[str, str]]:
    """
    Parse the read sequence and next genomic base from a Bismark XG-like tag.

    The original workflow expected a value formatted like 'prefix_cleanRead_nextContext'.
    This function preserves that behavior and returns None if parsing fails.
    """
    parts = str(xg_tag).split("_")
    if len(parts) < 3 or not parts[1] or not parts[2]:
        return None
    return parts[1], parts[2][0]


def read_methylation_ratio(xm_tag: str, min_cpg: int = 3) -> Optional[float]:
    """Calculate X/(X+x) from a Bismark XM tag when at least min_cpg CpGs are present."""
    methylated = xm_tag.count("X")
    unmethylated = xm_tag.count("x")
    total = methylated + unmethylated
    if total < min_cpg:
        return None
    return round(methylated / total, 2)


def extract_reads_for_bed(
    bed_file: Path,
    bam_files: Iterable[Path],
    dmr_bounds: dict[str, tuple[float, float]],
    outdir: Path,
    min_cpg: int = 3,
    skip_insertions: bool = True,
    skip_ambiguous_reads: bool = True,
) -> list[Path]:
    """Extract reads overlapping one BED file from all BAM files."""
    dmr_name = bed_file.stem
    read_outdir = outdir / dmr_name / "sample_all_switch_reads"
    read_outdir.mkdir(parents=True, exist_ok=True)

    bed_df = pd.read_csv(bed_file, sep="\t", header=None, names=["chr", "start", "end"])
    output_files: list[Path] = []

    for bam_file in bam_files:
        output_file = read_outdir / f"{bam_file.stem}_{dmr_name}_reads.txt"
        output_files.append(output_file)
        logging.info("Extracting reads from %s using %s", bam_file.name, bed_file.name)

        with pysam.AlignmentFile(bam_file, "rb") as bam, output_file.open("w") as out_handle:
            for row in bed_df.itertuples(index=False):
                chrom = str(row.chr)
                start = int(row.start)
                end = int(row.end)
                target_key = f"{chrom}_{start}_{end}"
                if target_key not in dmr_bounds:
                    logging.warning("Skipping %s because it was not found in the DMR bounds table.", target_key)
                    continue
                ref_min, _ref_max = dmr_bounds[target_key]

                for read in bam.fetch(chrom, start, end):
                    if skip_insertions and read.cigarstring and "I" in read.cigarstring:
                        continue
                    try:
                        xg_tag = read.get_tag("XG")
                        xm_tag = read.get_tag("XM")
                    except KeyError:
                        continue
                    if skip_ambiguous_reads and "N" in str(xg_tag):
                        continue

                    ratio = read_methylation_ratio(str(xm_tag), min_cpg=min_cpg)
                    if ratio is None:
                        continue

                    out_handle.write(
                        "\t".join(
                            [
                                target_key,
                                str(read.reference_name),
                                str(read.reference_start),
                                str(read.reference_end),
                                str(xg_tag),
                                str(xm_tag),
                                f"{ratio:.2f}",
                                str(ref_min),
                            ]
                        )
                        + "\n"
                    )
    return output_files


def best_cpg_window(seq: str, next_base: str, win: int = 65) -> int:
    """Return the 0-based start of the fixed-length window with the largest CpG count."""
    length = len(seq)
    if length <= win:
        return 0

    cpg = [0] * max(0, length - 1)
    for i in range(length - 1):
        if seq[i] == "C" and seq[i + 1] == "G":
            cpg[i] = 1

    prefix = [0] * (len(cpg) + 1)
    for i, value in enumerate(cpg):
        prefix[i + 1] = prefix[i] + value

    best_start = 0
    best_score = -1
    for start in range(0, length - win + 1):
        left = start
        right = start + win - 1
        score = prefix[right] - prefix[left]
        if start + win == length and seq[-1] == "C" and next_base == "G":
            score += 1
        if score > best_score:
            best_score = score
            best_start = start
    return best_start


def build_binary_methylation(clean_read: str, next_base: str, xm_tag: str) -> str:
    """Build a binary methylation string where methylated CpGs are coded as 1."""
    methylation = []
    max_len = min(len(clean_read), len(xm_tag))
    for i in range(max_len):
        if xm_tag[i] != "X":
            methylation.append("0")
            continue
        if i < max_len - 1:
            methylation.append("1" if clean_read[i] == "C" and clean_read[i + 1] == "G" else "0")
        else:
            methylation.append("1" if clean_read[i] == "C" and next_base == "G" else "0")
    return "".join(methylation)


def encode_read_file(read_file: Path, window_size: int = 65) -> tuple[Path, int, int]:
    """Convert extracted read records into fixed-length methylation encodings."""
    output_file = read_file.with_name(read_file.name.replace(".txt", "_clean_reads.txt"))
    total_reads = 0
    retained_reads = 0
    chr_mapping = {"chrX": "chr23", "chrY": "chr24"}

    with read_file.open("r") as in_handle, output_file.open("w") as out_handle:
        for line in in_handle:
            total_reads += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            _target_key, chrom, start, _end, xg_tag, xm_tag, _ratio, _cutoff = fields[:8]
            parsed = parse_read_sequence_from_xg(xg_tag)
            if parsed is None:
                continue
            clean_read, next_base = parsed
            methylation = build_binary_methylation(clean_read, next_base, xm_tag)
            if len(methylation) < window_size:
                continue

            window_start = best_cpg_window(clean_read, next_base, win=window_size)
            window_end = window_start + window_size
            out_read = clean_read[window_start:window_end]
            out_methylation = methylation[window_start:window_end]

            out_chrom = chr_mapping.get(chrom, chrom)
            try:
                out_start = str(int(start) + window_start)
            except ValueError:
                out_start = start

            out_handle.write("\t".join([out_chrom, out_start, out_read, out_methylation]) + "\n")
            retained_reads += 1

    return output_file, total_reads, retained_reads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract DMR-overlapping BAM reads and encode fixed-length methylation patterns."
    )
    parser.add_argument("--dmr-file", required=True, type=Path, help="DMR statistics table with keys/ref_min/ref_max/region columns.")
    parser.add_argument("--bam-dir", required=True, type=Path, help="Directory containing input BAM files.")
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--bam-pattern", default="*.bam", help="Glob pattern for input BAM files. Default: *.bam")
    parser.add_argument("--dmr-sets", default=DEFAULT_DMR_SETS, help="Comma-separated DMR sets, e.g. hypo_100,hyper_500.")
    parser.add_argument("--bed-dir", type=Path, default=None, help="Optional directory containing existing BED files. If omitted, BEDs are exported from --dmr-file.")
    parser.add_argument("--window-size", type=int, default=65, help="Fixed read window length used for encoding. Default: 65")
    parser.add_argument("--min-cpg", type=int, default=3, help="Minimum CpG count required per read. Default: 3")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip BAM extraction and only encode existing *_reads.txt files under --outdir.")
    parser.add_argument("--skip-encoding", action="store_true", help="Skip fixed-length read encoding after extraction.")
    parser.add_argument("--overwrite-bed", action="store_true", help="Overwrite exported BED files if they already exist.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress messages.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(levelname)s] %(message)s",
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    dmr_df = load_dmr_table(args.dmr_file)
    dmr_bounds = build_dmr_bounds(dmr_df)

    if args.bed_dir is not None:
        bed_files = sorted(args.bed_dir.glob("*/*.bed")) + sorted(args.bed_dir.glob("*.bed"))
        if not bed_files:
            raise FileNotFoundError(f"No BED files were found in {args.bed_dir}")
    else:
        dmr_sets = parse_dmr_sets(args.dmr_sets)
        ranked_beds = build_ranked_dmr_tables(dmr_df)
        bed_files = export_top_dmr_beds(ranked_beds, dmr_sets, args.outdir, overwrite=args.overwrite_bed)

    read_files: list[Path] = []
    if not args.skip_extraction:
        bam_files = find_bam_files(args.bam_dir, pattern=args.bam_pattern)
        for bed_file in bed_files:
            read_files.extend(
                extract_reads_for_bed(
                    bed_file=bed_file,
                    bam_files=bam_files,
                    dmr_bounds=dmr_bounds,
                    outdir=args.outdir,
                    min_cpg=args.min_cpg,
                )
            )
    else:
        read_files = sorted(args.outdir.glob("*/*/*_reads.txt"))

    if not args.skip_encoding:
        for read_file in read_files:
            if read_file.name.endswith("_clean_reads.txt"):
                continue
            output_file, total_reads, retained_reads = encode_read_file(read_file, window_size=args.window_size)
            ratio = retained_reads / total_reads if total_reads else 0.0
            logging.info(
                "Encoded %s -> %s: total=%d retained=%d retained_fraction=%.4f",
                read_file.name,
                output_file.name,
                total_reads,
                retained_reads,
                ratio,
            )


if __name__ == "__main__":
    main()
