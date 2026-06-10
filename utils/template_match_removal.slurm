#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1

echo "Running on node: $HOSTNAME"

JSONL_PATH=$1
OUTPUT_DIR=$2
NUMWORKERS=$3
INPUT_MATCHED_NGRAMS_DIR=$4
MATCH_THRESHOLD=$5
PARSE_JSONL=$6
TMP_DIR=$7
DATASETS_DIR=$8
METADATA_PATH=$9

LINENUMBER=$SLURM_ARRAY_TASK_ID

# Read that specific line
line=$(sed -n "${LINENUMBER}p" "$JSONL_PATH")
echo "Reading line number $LINENUMBER: $line"

# Extract fields using jq
INPUT_FILE_TXT=$(echo "$line" | jq -r ".path")
INPUT_TASK_NGRAMS=$(echo "$line" | jq -r ".task_benchmark_ngrams")
JOB_NAME=$(echo "$line" | jq -r ".name")
OUTPUT_REMOVED="${OUTPUT_DIR}"

srun /usr/bin/time -v singularity exec \
    --bind /appl/local/openeurollm/training/catalogue:/appl/local/openeurollm/training/catalogue \
    --bind $INPUT_MATCHED_NGRAMS_DIR:/matched_ngrams \
    --bind $TMP_DIR:/tmp_nemo \
    --bind $DATASETS_DIR:/data \
    --bind $OUTPUT_DIR:/removed \
    nemo.sif \
    python3 remove_matching_ngrams_parallel.py \
    --input-data-file $INPUT_FILE_TXT \
    --input-matched-ngrams /matched_ngrams/all_matched_ngrams.pkl \
    --output-removed-doc-dir "/removed" \
    --n-workers $NUMWORKERS \
    --max-document-splits 1 \
    --match-threshold $MATCH_THRESHOLD \
    --metadata-path "$METADATA_PATH" \
    $PARSE_JSONL
