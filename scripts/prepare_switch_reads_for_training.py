#!/usr/bin/env python3
"""
Prepare direction-specific switch reads for model training.

This script builds ranked hypo-/hyper-DMR BED files from a DMR summary table,
extracts direction-specific reads from tumor and reference BAM files, and encodes
reads into fixed-length binary methylation strings for downstream model training.

Default behavior is designed to preserve the logic of the original analysis:
- hypo DMRs are ranked by below_ratio in descending order;
- hyper DMRs are ranked by above_ratio in descending order;
- tumor reads are selected when they are abnormal relative to the reference bound;
- reference reads are selected in the opposite direction;
- encoded reads are standardized to 140 bp.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import pysam


DEFAULT_DMR_SETS = (
    "hypo_100,hypo_200,hypo_500,hypo_1000,hypo_2000,hypo_3000,hypo_4000,hypo_5000,"
    "hyper_100,hyper_200,hyper_500,hyper_1000,hyper_1500,hyper_2000,hyper_3000"
)

REQUIRED_COLUMNS = {
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
}


@dataclass(frozen=True)
class DMRSet:
    direction: str
    top_n: int

    @property
    def name(self) -> str:
        return f"{self.direction}_{self.top_n}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate ranked DMR BED files, extract direction-specific reads from "
            "tumor/reference BAM files, and encode reads into fixed-length binary "
            "methylation strings."
        )
    )
    parser.add_argument("--dmr-file", required=True, help="DMR summary table with keys/ref_min/ref_max/region/ratio columns.")
    parser.add_argument("--ref-bam-dir", required=True, help="Directory containing reference/control BAM files.")
    parser.add_argument("--tumor-bam-dir", required=True, help="Directory containing tumor BAM files.")
    parser.add_argument("--outdir", default="switch_read_training_data", help="Output directory.")
    parser.add_argument("--dmr-sets", default=DEFAULT_DMR_SETS, help="Comma-separated DMR sets, e.g. hypo_100,hyper_1000.")
    parser.add_argument("--bam-pattern", default="*.bam", help="Glob pattern for BAM files.")
    parser.add_argument("--window-size", type=int, default=140, help="Encoded read length. Default: 140.")
    parser.add_argument("--min-cpg", type=int, default=3, help="Minimum CpG observations per read. Default: 3.")
    parser.add_argument("--xg-tag", default="XG", help="BAM tag containing read sequence/context string. Default: XG.")
    parser.add_argument("--xm-tag", default="XM", help="BAM tag containing Bismark methylation string. Default: XM.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing read and encoded files.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress information.")
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="[%(levelname)s] %(message)s",
    )


def parse_dmr_sets(dmr_sets: str) -> list[DMRSet]:
    parsed: list[DMRSet] = []
    for item in [x.strip() for x in dmr_sets.split(",") if x.strip()]:
        try:
            direction, number = item.split("_", 1)
            if direction not in {"hypo", "hyper"}:
                raise ValueError
            parsed.append(DMRSet(direction=direction, top_n=int(number)))
        except ValueError as exc:
            raise ValueError(f"Invalid DMR set '{item}'. Expected format hypo_100 or hyper_100.") from exc
    if not parsed:
        raise ValueError("No DMR sets were provided.")
    return parsed


def load_dmr_table(dmr_file: Path) -> pd.DataFrame:
    if not dmr_file.exists():
        raise FileNotFoundError(f"DMR file not found: {dmr_file}")
    df = pd.read_csv(dmr_file, sep="\t")
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"DMR file is missing required columns: {sorted(missing)}")
    df = df[list(REQUIRED_COLUMNS)].copy()
    return df


def split_keys_to_bed(keys: pd.Series) -> pd.DataFrame:
    bed = keys.astype(str).str.split("_", expand=True)
    if bed.shape[1] != 3:
        raise ValueError("The 'keys' column must have format chr_start_end.")
    bed.columns = ["chr", "st", "end"]
    return bed


def build_ranked_dmr_tables(dmr_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    hypo = dmr_df[dmr_df["region"].astype(str).str.contains("hypo", case=False, na=False)].copy()
    hypo.sort_values("below_ratio", ascending=False, inplace=True)

    hyper = dmr_df[dmr_df["region"].astype(str).str.contains("hyper", case=False, na=False)].copy()
    hyper.sort_values("above_ratio", ascending=False, inplace=True)

    return {"hypo": hypo, "hyper": hyper}


def export_top_dmr_bed(ranked: dict[str, pd.DataFrame], dmr_set: DMRSet, outdir: Path) -> Path:
    set_dir = outdir / dmr_set.name
    set_dir.mkdir(parents=True, exist_ok=True)
    source = ranked[dmr_set.direction].head(dmr_set.top_n)
    bed = split_keys_to_bed(source["keys"])
    bed_path = set_dir / f"{dmr_set.name}.bed"
    bed.to_csv(bed_path, sep="\t", header=False, index=False)
    return bed_path


def list_bams(directory: Path, pattern: str) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"BAM directory not found: {directory}")
    bams = sorted(directory.glob(pattern))
    if not bams:
        raise FileNotFoundError(f"No BAM files found in {directory} using pattern '{pattern}'.")
    return bams


def get_read_tag(read: pysam.AlignedSegment, tag: str) -> str | None:
    try:
        return read.get_tag(tag)
    except KeyError:
        return None


def methylation_ratio(xm_string: str) -> float | None:
    unmethylated = xm_string.count("x")
    methylated = xm_string.count("X")
    total = methylated + unmethylated
    if total < 1:
        return None
    return methylated / total


def extract_directional_reads(
    bam_files: Sequence[Path],
    bed_path: Path,
    dmr_df: pd.DataFrame,
    dmr_set: DMRSet,
    sample_class: str,
    output_dir: Path,
    min_cpg: int,
    xg_tag: str,
    xm_tag: str,
    overwrite: bool,
) -> list[Path]:
    """Extract reads passing direction-specific thresholds.

    For tumor samples:
      hypo: ratio < ref_min; hyper: ratio > ref_max.
    For reference samples:
      hypo: ratio > ref_min; hyper: ratio < ref_max.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    bed_df = pd.read_csv(bed_path, sep="\t", header=None, names=["chr", "st", "end"])
    ref_bounds = dmr_df.set_index("keys")[["ref_min", "ref_max"]].to_dict("index")
    written_files: list[Path] = []

    for bam_path in bam_files:
        prefix = "tumor" if sample_class == "tumor" else "ref"
        out_file = output_dir / f"{prefix}_{bam_path.name.replace('.bam', f'_{dmr_set.name}_reads.txt')}"
        if out_file.exists() and not overwrite:
            logging.info("Skipping existing file: %s", out_file)
            written_files.append(out_file)
            continue

        n_written = 0
        with pysam.AlignmentFile(bam_path, "rb") as bam, out_file.open("w") as out_handle:
            for _, row in bed_df.iterrows():
                chrom = str(row["chr"])
                start = int(row["st"])
                end = int(row["end"])
                target_key = f"{chrom}_{start}_{end}"
                if target_key not in ref_bounds:
                    continue

                ref_min = float(ref_bounds[target_key]["ref_min"])
                ref_max = float(ref_bounds[target_key]["ref_max"])

                for read in bam.fetch(chrom, start, end):
                    cigar = read.cigarstring or ""
                    if "I" in cigar:
                        continue

                    read_context = get_read_tag(read, xg_tag)
                    xm_string = get_read_tag(read, xm_tag)
                    if read_context is None or xm_string is None:
                        continue
                    if "N" in read_context:
                        continue

                    unmethylated = xm_string.count("x")
                    methylated = xm_string.count("X")
                    if methylated + unmethylated < min_cpg:
                        continue

                    ratio = round(methylated / (methylated + unmethylated), 2)
                    if dmr_set.direction == "hypo":
                        cutoff = ref_min
                        keep = ratio < ref_min if sample_class == "tumor" else ratio > ref_min
                    else:
                        cutoff = ref_max
                        keep = ratio > ref_max if sample_class == "tumor" else ratio < ref_max

                    if not keep:
                        continue

                    out_handle.write(
                        f"{target_key}\t{read.reference_name}\t{read.reference_start}\t{read.reference_end}\t"
                        f"{read_context}\t{xm_string}\t{ratio}\t{cutoff}\n"
                    )
                    n_written += 1

        logging.info("Wrote %d reads: %s", n_written, out_file)
        written_files.append(out_file)

    return written_files


def parse_clean_read_and_next_base(read_context: str) -> tuple[str, str] | None:
    parts = read_context.split("_")
    if len(parts) < 3 or not parts[1] or not parts[2]:
        return None
    return parts[1], parts[2][0]


def build_binary_methylation(clean_read: str, next_base: str, xm_string: str) -> str:
    methylation = []
    max_len = min(len(clean_read), len(xm_string))
    for i in range(max_len):
        if xm_string[i] == "X":
            if i < max_len - 1:
                methylation.append("1" if clean_read[i] == "C" and clean_read[i + 1] == "G" else "0")
            else:
                methylation.append("1" if clean_read[i] == "C" and next_base == "G" else "0")
        else:
            methylation.append("0")
    return "".join(methylation)


def standardize_window(sequence: str, methylation: str, window_size: int) -> tuple[str, str, int] | None:
    length = len(methylation)
    if length < window_size:
        return None
    if length == window_size:
        return sequence[:window_size], methylation[:window_size], 0
    if length == 150 and window_size == 140:
        return sequence[5:145], methylation[5:145], 5

    diff_len = length - window_size
    clip = diff_len // 2
    return sequence[clip : clip + window_size], methylation[clip : clip + window_size], clip


def normalize_chromosome_name(chrom: str) -> str:
    return {"chrX": "chr23", "chrY": "chr24"}.get(chrom, chrom)


def encode_read_file(read_file: Path, window_size: int, overwrite: bool) -> Path:
    out_file = read_file.with_name(read_file.name.replace(".txt", "_clean_reads.txt"))
    if out_file.exists() and not overwrite:
        logging.info("Skipping existing encoded file: %s", out_file)
        return out_file

    total_reads = 0
    kept_reads = 0
    with read_file.open("r") as in_handle, out_file.open("w") as out_handle:
        for line in in_handle:
            total_reads += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            _, chrom, start, _, read_context, xm_string, _, _ = fields[:8]
            parsed = parse_clean_read_and_next_base(read_context)
            if parsed is None:
                continue
            clean_read, next_base = parsed
            methylation = build_binary_methylation(clean_read, next_base, xm_string)
            standardized = standardize_window(clean_read, methylation, window_size)
            if standardized is None:
                continue
            out_sequence, out_methylation, offset = standardized
            try:
                adjusted_start = str(int(start) + offset)
            except ValueError:
                adjusted_start = start
            out_handle.write(
                f"{normalize_chromosome_name(chrom)}\t{adjusted_start}\t{out_sequence}\t{out_methylation}\n"
            )
            kept_reads += 1

    ratio = kept_reads / total_reads if total_reads else 0.0
    logging.info("Encoded %s: total=%d, kept=%d, fraction=%.4f", read_file.name, total_reads, kept_reads, ratio)
    return out_file


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dmr_sets = parse_dmr_sets(args.dmr_sets)
    dmr_df = load_dmr_table(Path(args.dmr_file))
    ranked = build_ranked_dmr_tables(dmr_df)

    ref_bams = list_bams(Path(args.ref_bam_dir), args.bam_pattern)
    tumor_bams = list_bams(Path(args.tumor_bam_dir), args.bam_pattern)

    all_read_files: list[Path] = []
    for dmr_set in dmr_sets:
        bed_path = export_top_dmr_bed(ranked, dmr_set, outdir)
        set_dir = outdir / dmr_set.name
        logging.info("Processing DMR set: %s", dmr_set.name)

        all_read_files.extend(
            extract_directional_reads(
                bam_files=tumor_bams,
                bed_path=bed_path,
                dmr_df=dmr_df,
                dmr_set=dmr_set,
                sample_class="tumor",
                output_dir=set_dir / "tumor_switch_reads",
                min_cpg=args.min_cpg,
                xg_tag=args.xg_tag,
                xm_tag=args.xm_tag,
                overwrite=args.overwrite,
            )
        )
        all_read_files.extend(
            extract_directional_reads(
                bam_files=ref_bams,
                bed_path=bed_path,
                dmr_df=dmr_df,
                dmr_set=dmr_set,
                sample_class="ref",
                output_dir=set_dir / "ref_switch_reads",
                min_cpg=args.min_cpg,
                xg_tag=args.xg_tag,
                xm_tag=args.xm_tag,
                overwrite=args.overwrite,
            )
        )

    for read_file in all_read_files:
        encode_read_file(read_file, window_size=args.window_size, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
