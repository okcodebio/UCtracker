#!/usr/bin/env bash
set -euo pipefail

python scripts/prepare_switch_reads_for_training.py \
  --dmr-file reference/dmr_ranked_regions.txt \
  --ref-bam-dir bam/reference \
  --tumor-bam-dir bam/tumor \
  --outdir switch_read_training_data \
  --dmr-sets hypo_100,hypo_200,hypo_500,hypo_1000,hypo_2000,hyper_100,hyper_500,hyper_1000 \
  --window-size 140 \
  --min-cpg 3 \
  --verbose
