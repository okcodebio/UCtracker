#!/usr/bin/env bash
set -euo pipefail

python scripts/extract_and_encode_switch_reads.py \
  --dmr-file reference/dmr_ranked_regions.txt \
  --bam-dir bam \
  --outdir switch_read_features \
  --dmr-sets hypo_100,hypo_200,hypo_500,hypo_1000,hypo_2000,hyper_100,hyper_500,hyper_1000 \
  --window-size 65 \
  --min-cpg 3 \
  --verbose
