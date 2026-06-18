#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1

echo "Running on node: $HOSTNAME"

JSONL_PATH=$1
OUTPUT_MATCHED_NGRAMS_DIR=$2
NUMWORKERS=$3
PARSE_JSONL=$4
TMP_DIR=$5
DATASETS_DIR=$6
METADATA_PATH=$7

LINENUMBER=$SLURM_ARRAY_TASK_ID

# Read that specific line
line=$(sed -n "${LINENUMBER}p" "$JSONL_PATH")
echo "Reading line number $LINENUMBER: $line"

# Extract fields using jq
INPUT_FILE_TXT=$(echo "$line" | jq -r ".path")
INPUT_TASK_NGRAMS=$(echo "$line" | jq -r ".task_benchmark_ngrams")

srun /usr/bin/time -v singularity exec \
    --bind /appl/local/openeurollm/training/catalogue:/appl/local/openeurollm/training/catalogue \
    --bind $OUTPUT_MATCHED_NGRAMS_DIR:/matched_ngrams_dir \
    --bind $TMP_DIR:/tmp_nemo \
    --bind $DATASETS_DIR:/data \
    /scratch/project_465002530/training/collection/flag/nemo-curator/nemo.sif \
    python3 find_matching_ngrams.py \
    --input-data-file=$INPUT_FILE_TXT \
    --input-task-ngrams $INPUT_TASK_NGRAMS \
    --output-matched-ngram-dir /matched_ngrams_dir  \
    --n-workers $NUMWORKERS \
    --metadata-path "$METADATA_PATH" \
    $PARSE_JSONL
