#!/usr/bin/env bash
set -euo pipefail

python ../scripts/summarize_bin_features.py \
  --input-dir ./bin_feature_files \
  --output-prefix ./results/merged_bin_features \
  --file-keyword cts_aver_mhl_ts_4cpg \
  --verbose
