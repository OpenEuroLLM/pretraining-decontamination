import os
from pathlib import Path
import subprocess
import yaml

with open('env_variables.yaml', 'r') as file:
    variables = yaml.safe_load(file)
    PROJECT_ID = variables["PROJECT_ID"]

LOGS_FOLDER=Path("logs", "benchmark_ngrams")
Path(LOGS_FOLDER).mkdir(exist_ok=True, parents=True)


for file in os.listdir("benchmarks"):
    if not file.endswith(".yaml"):
        continue
    command = f"HF_TOKEN=XXX srun --job-name=nemo-curator --cpus-per-task=12 --ntasks=1 --nodes=1 --time=15 --mem=50G --partition=debug --account={PROJECT_ID} singularity exec nemo.sif python3 prepare_task_data.py --task-config-file=benchmarks/{file} --output-task-ngrams=1_task_ngrams/{Path(file).stem}.pkl --memory_limit=50GB --n_workers=12 > {LOGS_FOLDER}/{Path(file).stem}.log 2>&1"
    print(command)
    subprocess.run(command, shell=True)

# LANG="english"
# command = f"HF_TOKEN=XXX srun --job-name=nemo-curator --cpus-per-task=12 --ntasks=1 --nodes=1 --time=15 --mem=50G --partition=debug --account={PROJECT_ID} singularity exec nemo.sif python3 prepare_task_data.py --task-config-file=benchmarks/{LANG}_benchmarks.yaml --output-task-ngrams=1_task_ngrams/{LANG}_benchmarksv2.pkl --memory_limit=50GB --n_workers=12 > {LOGS_FOLDER}/{LANG}_benchmarks.log 2>&1"
# subprocess.run(command, shell=True)
