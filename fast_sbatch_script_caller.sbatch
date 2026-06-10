#!/bin/bash
#SBATCH --job-name=rsync_copy_job
#SBATCH --nodes=1
#SBATCH --partition=small
#SBATCH --cpus-per-task=128
#SBATCH --time=05:30:00
#SBATCH --mem=440G
#SBATCH --account=project_465002530
#SBATCH --output=/scratch/project_462000963/users/tudormateiu/tmp/general_logs/RSYNC_COPY_%j.out
#SBATCH --error=/scratch/project_462000963/users/tudormateiu/tmp/general_logs/DECONT_fast_sbatch_%j.err

set -euo pipefail

bash fast_script_caller.sh

echo "Done."
