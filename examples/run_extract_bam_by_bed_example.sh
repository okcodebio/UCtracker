#!/usr/bin/env bash
set -euo pipefail

python scripts/extract_bam_by_bed.py \
  --bed reference/target_regions.bed \
  --bam-dir bam \
  --outdir target_bam \
  --processes 8 \
  --verbose
