# UCtracker

UCtracker is a lightweight Python toolkit for extracting and summarizing read-level DNA methylation features from bisulfite-aligned BAM files.

The toolkit was prepared for reproducible research and public code release. Local paths, user-specific server information, and project-specific identifiers have been removed. All input files are provided through command-line arguments.

## Main functions

1. `methyl_read_bin_profiler.py`  
   Extracts per-read methylation features from BAM files within fixed-width genomic bins.

2. `summarize_bin_features.py`  
   Aggregates per-read feature values across samples and calculates per-bin summary statistics for:
   - average methylation level
   - methylation haplotype load (MHL)
   - methylation transition score

3. `calculate_abnormal_read_fraction.py`  
   Calculates the fraction of reads with methylation values above or below reference boundaries in selected DMRs.

4. `extract_bam_by_bed.py`  
   Batch-extracts reads from BAM files overlapping a BED file and creates indexed region-specific BAM files.

## Installation

```bash
conda create -n UCtracker python=3.10 -y
conda activate UCtracker
pip install -r requirements.txt
```

## Requirements

- Python >= 3.8
- numpy
- pandas
- pysam
- pyfaidx
- samtools, available in `PATH` or provided with `--samtools`

## 1. Extract read-level methylation features

```bash
python scripts/methyl_read_bin_profiler.py \
  --bam sample.bam \
  --reference hg19.fa \
  --outdir results \
  --bin-size 500
```

For maximum compatibility with the original internal script, the legacy BAM tag index can be used:

```bash
python scripts/methyl_read_bin_profiler.py \
  --bam sample.bam \
  --reference hg19.fa \
  --outdir results \
  --bin-size 500 \
  --legacy-tag-index 3
```

## 2. Summarize bin-level features across samples

```bash
python scripts/summarize_bin_features.py \
  --input-dir results \
  --output-prefix merged_bin_features \
  --file-keyword cts_aver_mhl_ts_4cpg \
  --verbose
```

This generates one output file for each feature type:

```text
merged_bin_features_average.txt
merged_bin_features_mhl.txt
merged_bin_features_transitions_score.txt
```

Each output row corresponds to one genomic bin. The columns are:

```text
chromosome    start    end    min    median    mean    max    std    outlier_number    outlier_values    count_after_filtering
```


## 3. Calculate abnormal read fractions in selected DMRs

```bash
python scripts/calculate_abnormal_read_fraction.py \
  --feature-file results/sample_500_aver_mhl_ts_3cpg.txt \
  --dmr-file reference/dmr_reference_bounds.txt \
  --feature average \
  --output results/sample_abnormal_reads.txt \
  --verbose
```

The DMR/reference file should contain chromosome, start, and end in the first three columns, and must include the following columns:

```text
ref_min    ref_max
```

For each matched region, the script reports:

```text
total    count_above    above_ratio    count_below    below_ratio
```

This script is useful for estimating the fraction of abnormal methylation reads in predefined DMRs. By default, it uses the `average` read-level methylation feature, matching the original internal script.

## 4. Extract target-region BAM files using a BED file

```bash
python scripts/extract_bam_by_bed.py \
  --bed reference/target_regions.bed \
  --bam-dir bam \
  --outdir target_bam \
  --processes 8 \
  --verbose
```

By default, each input BAM file is converted to a region-specific BAM file using the original naming pattern:

```text
<input_basename>.switch.bam
<input_basename>.switch.bam.bai
```

The script internally runs the equivalent of:

```bash
samtools view -b -h input.bam -L target_regions.bed > output.switch.bam
samtools index output.switch.bam
```

Compared with the original internal script, shell execution has been replaced with `subprocess` calls using argument lists, which avoids shell-quoting problems and reduces command-injection risk. Existing output BAM files are skipped unless `--overwrite` is provided.


## Output compatibility

The output formats were intentionally kept consistent with the original analysis scripts to avoid changing downstream results. The main changes are code organization, command-line configuration, error handling, and removal of local sensitive paths.

## Notes

- BAM files should be coordinate-sorted and indexed.
- The default methylation tag is `XM`, which is commonly produced by Bismark.
- Reference genome FASTA files should be indexed or indexable by `pyfaidx`.
- No model weights, patient information, sample identifiers, or server paths are included in this repository.

## Suggested repository name

`MethylReadBinProfiler`

## License

Add a license according to your publication or institutional policy.

## 5. Extract and encode DMR-overlapping switch reads

```bash
python scripts/extract_and_encode_switch_reads.py \
  --dmr-file reference/dmr_ranked_regions.txt \
  --bam-dir bam \
  --outdir switch_read_features \
  --dmr-sets hypo_100,hypo_200,hypo_500,hypo_1000,hypo_2000,hyper_100,hyper_500,hyper_1000 \
  --window-size 65 \
  --min-cpg 3 \
  --verbose
```

The DMR table must include the following columns:

```text
keys    ref_min    ref_max    region    above_ratio    below_ratio
```

where `keys` should use the format:

```text
chr_start_end
```

The script first ranks hypo-DMRs by `below_ratio` and hyper-DMRs by `above_ratio`, exports the requested top-ranked DMR sets as BED files, extracts overlapping reads from BAM files, and then encodes each retained read into a fixed-length binary methylation string.

For each DMR set, the intermediate read file is written as:

```text
<dmr_set>/sample_all_switch_reads/<sample>_<dmr_set>_reads.txt
```

The encoded fixed-length output is written as:

```text
<dmr_set>/sample_all_switch_reads/<sample>_<dmr_set>_reads_clean_reads.txt
```

The encoded output columns are:

```text
chromosome    adjusted_start    read_sequence_65bp    binary_methylation_65bp
```

`chrX` and `chrY` are converted to `chr23` and `chr24`, respectively, to preserve compatibility with numeric chromosome-based downstream modeling.

### 6. Prepare direction-specific switch reads for model training

`prepare_switch_reads_for_training.py` generates ranked hypo-/hyper-DMR BED files, extracts direction-specific reads from tumor and reference BAM files, and encodes reads into fixed-length binary methylation strings. By default, hypo DMRs are ranked by `below_ratio`, hyper DMRs are ranked by `above_ratio`, tumor reads are selected when they are abnormal relative to the reference bound, and reference reads are selected in the opposite direction.

Example:

```bash
python scripts/prepare_switch_reads_for_training.py \
  --dmr-file reference/dmr_ranked_regions.txt \
  --ref-bam-dir bam/reference \
  --tumor-bam-dir bam/tumor \
  --outdir switch_read_training_data \
  --dmr-sets hypo_100,hypo_200,hypo_500,hypo_1000,hypo_2000,hyper_100,hyper_500,hyper_1000 \
  --window-size 140 \
  --min-cpg 3 \
  --verbose
```

Main outputs:

```text
<dmr_set>/<dmr_set>.bed
<dmr_set>/tumor_switch_reads/tumor_<sample>_<dmr_set>_reads.txt
<dmr_set>/ref_switch_reads/ref_<sample>_<dmr_set>_reads.txt
<dmr_set>/tumor_switch_reads/tumor_<sample>_<dmr_set>_reads_clean_reads.txt
<dmr_set>/ref_switch_reads/ref_<sample>_<dmr_set>_reads_clean_reads.txt
```

The encoded files contain four tab-delimited columns: chromosome, adjusted start coordinate, fixed-length read sequence, and fixed-length binary methylation encoding.

## 7. Train the read-level CNN-BiLSTM classifier

`train_read_methylation_cnn_bilstm.py` trains the read-level deep learning classifier using the fixed-length files generated by `prepare_switch_reads_for_training.py`. The default architecture and hyperparameters preserve the original analysis script: Conv1D, MaxPooling1D, Dropout, Bidirectional LSTM, Conv1D, MaxPooling1D, Dropout, Flatten, Dense(750), Dropout, Dense(300), and a sigmoid output layer.

Example:

```bash
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
```

The input directory should contain:

```text
ref_switch_reads/*_reads_clean_reads.txt
tumor_switch_reads/*_reads_clean_reads.txt
```

The encoded read files must contain four tab-delimited columns:

```text
chromosome    start    read_sequence    binary_methylation
```

Main outputs are written to `train_out/` by default and keep the original filenames for downstream compatibility:

```text
weight.h5
predict_result.txt
test_label.txt
test_chrom.txt
test_region.txt
result_all.txt
label_all.txt
chrom_all.txt
region_all.txt
data_lstm.txt
data_origin_lstm.txt
label_origin.txt
chrom_origin.txt
region_origin.txt
```

The script provides an optional `--seed` argument for reproducible public runs. If `--seed` is omitted, stochastic behavior remains consistent with the original internal script.
