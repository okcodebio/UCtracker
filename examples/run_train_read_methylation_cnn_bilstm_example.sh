#!/usr/bin/env bash
set -euo pipefail

python scripts/train_read_methylation_cnn_bilstm.py \
  --input-dir switch_read_training_data/hypo_2000 \
  --outdir switch_read_training_data/hypo_2000/train_out \
  --read-length 140 \
  --min-cpg 3 \
  --epochs 150 \
  --batch-size 128 \
  --learning-rate 1e-4 \
  --validation-split 0.1 \
  --verbose
