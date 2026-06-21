#!/bin/bash
set -euo pipefail

SIF=./nemo.sif
PROJ=/scratch/project_465002530
HFHOME=$PROJ/users/tudormateiu/.cache/huggingface

singularity exec \
  --bind "$PROJ:$PROJ" \
  --bind "$HOME:$HOME" \
  --env HF_HOME="$HFHOME" \
  --env HF_TOKEN="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" \
  --pwd "$PWD" \
  "$SIF" \
  python3 check_benchmark_ngrams.py benchmarks/english_benchmarks.yaml
