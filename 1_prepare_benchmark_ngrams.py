import os
from pathlib import Path
import subprocess
import yaml
from loguru import logger

import os
os.environ["HF_TOKEN"] = "xxx"
os.environ["HF_HOME"] = "/scratch/project_465002530/users/tudormateiu/.cache/huggingface"

with open('env_variables.yaml', 'r') as file:
    variables = yaml.safe_load(file)
    PROJECT_ID = variables["PROJECT_ID"]

LOGS_FOLDER=Path("logs", "benchmark_ngrams")
Path(LOGS_FOLDER).mkdir(exist_ok=True, parents=True)

FAILURES = ["english"]

for file in os.listdir("benchmarks"):
    if not file.endswith(".yaml"):
        continue

    lang = Path(file).stem.replace("_benchmarks", "")

    if lang not in FAILURES:
        continue

    command = f"HF_TOKEN={os.environ['HF_TOKEN']} HF_HOME={os.environ['HF_HOME']} srun --job-name=nemo-curator --cpus-per-task=12 --ntasks=1 --nodes=1 --time=15 --mem=50G --partition=debug --account={PROJECT_ID} singularity exec --bind /scratch:/scratch,/pfs:/pfs nemo.sif python3 prepare_task_data.py --task-config-file=benchmarks/{file} --output-task-ngrams=1_task_ngrams/{Path(file).stem}.pkl --memory_limit=50GB --n_workers=12 > {LOGS_FOLDER}/{Path(file).stem}.log 2>&1"
    
    logger.info(f"[{lang}] Submitting job...")
    logger.debug(f"[{lang}] Command: {command}")
    
    subprocess.run(command, shell=True)
    
    logger.success(f"[{lang}] Job submitted.")
    #break



# LANG="english"
# print("RUNNING....")
# command = f"HF_TOKEN=xxx srun --job-name=nemo-curator --cpus-per-task=12 --ntasks=1 --nodes=1 --time=01:30:00 --mem=100G --partition=small --account={PROJECT_ID} singularity exec --bind /scratch:/scratch,/pfs:/pfs nemo.sif python3 prepare_task_data.py --task-config-file=benchmarks/{LANG}_benchmarks.yaml --output-task-ngrams=1_task_ngrams/{LANG}_benchmarksv2.pkl --memory_limit=50GB --n_workers=12 > {LOGS_FOLDER}/{LANG}_benchmarks.log 2>&1"
# subprocess.run(command, shell=True)
# print("FINISHED")

# command = f"HF_TOKEN={os.environ['HF_TOKEN']} HF_HOME={os.environ['HF_HOME']} srun --job-name=nemo-curator --cpus-per-task=128 --ntasks=1 --nodes=1 --time=1:00:00 --mem=220G --partition=small --account={PROJECT_ID} singularity exec --bind /scratch:/scratch,/pfs:/pfs nemo.sif python3 prepare_task_data.py --task-config-file=benchmarks/{LANG}_benchmarks.yaml --output-task-ngrams=1_task_ngrams/{LANG}_benchmarksv2.pkl --memory_limit=50GB --n_workers=12 > {LOGS_FOLDER}/{LANG}_benchmarks.log 2>&1"
