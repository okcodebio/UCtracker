#!/usr/bin/env bash
set -euo pipefail

python scripts/calculate_abnormal_read_fraction.py \
  --feature-file results/sample_500_aver_mhl_ts_3cpg.txt \
  --dmr-file reference/dmr_reference_bounds.txt \
  --feature average \
  --output results/sample_abnormal_reads.txt \
  --verbose
