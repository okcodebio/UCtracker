#!/usr/bin/env bash
set -euo pipefail

python scripts/methyl_read_bin_profiler.py \
  --bam /path/to/sample.bam \
  --reference /path/to/reference.fa \
  --outdir results \
  --bin-size 500
