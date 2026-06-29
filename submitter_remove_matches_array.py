
import os
import json
import shutil
import argparse
import subprocess
import polars as pl
import datetime
from pathlib import Path
from loguru import logger
import yaml


CODE_ELEGIBILITY = ["NOT_STARTED", "FAILED", "PREEMPTED", "SUSPENDED", None, "CANCELLED+", "TIMEOUT"]
WORK_DIRECTORY = None
with open('env_variables.yaml', 'r') as file:
    variables = yaml.safe_load(file)
    WORK_DIRECTORY = variables["DECONTAMINATION_DIR"]
    PROJECT_ID = variables["PROJECT_ID"]
    TMP_DIR = variables["TMP_DIR"]
    DATASETS_DIR = variables["DATASETS_DIR"]
    Path(WORK_DIRECTORY).mkdir(exist_ok=True, parents=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-jsonl", type=Path, required=True, help="JSONL file containing job information of dataset shards.")
    parser.add_argument("--node-limit", type=int, default=2, help="Limit on how many nodes (jobs) will be submitted.")
    parser.add_argument("--account", type=str, default=PROJECT_ID, help="Account where resources will be used.")
    parser.add_argument("--partition", type=str, required=True, help="Lumi partition to be used.")
    parser.add_argument("--time-limit", type=str, default="6:00:00", help="Time limit per job submitted")
    parser.add_argument("--n-workers", type=int, default=10, help="Number of workers NemoCurator will employ per job.")
    parser.add_argument("--memory", type=str, default="448G", help="Memory to use")
    parser.add_argument("--match-threshold", type=int, default="10", help="Memory to use")
    parser.add_argument("--parse-jsonl",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Flag used when jsonl fields differ in metadata type and pandas crashes.")
    args = parser.parse_args()

    x = datetime.datetime.now()
    Path("logs", "removing").mkdir(exist_ok=True, parents=True)
    log_name = f"logs/removing/finding_{args.shards_jsonl}_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log"
    logger.add(log_name)
    logger.info(f"_____________")
    logger.info(f"{log_name}")

    shards_dict = pl.read_ndjson(args.shards_jsonl).to_dicts()
    jobs_to_submit = []
    dataset_name = ""
    metadata_path = ""
    if not all([shard_group["status"] == "COMPLETED" for shard_group in shards_dict]):
        logger.error("Not all matching jobs have been completed. Exiting...")
        return
    for idx, shard_group in enumerate(shards_dict):
        if shard_group["status"] == "COMPLETED":
            actual_idx = str(idx + 1)
            dataset_name = shard_group["dataset_name"]
            metadata_path = shard_group.get("metadata_path") or ""
            # If remove job hasn't been started before
            if "removal_status" not in shard_group:
                jobs_to_submit.append(actual_idx)
            else:
                # If remove job has been started before and crashed
                if shard_group["removal_status"] in CODE_ELEGIBILITY:
                    jobs_to_submit.append(actual_idx)

            if len(jobs_to_submit) >= args.node_limit:
                break
    # Write back the jsonl with the updated shards that have started
    shards_df = pl.DataFrame(shards_dict)
    shards_df.write_ndjson(args.shards_jsonl)


    logs_path = Path(WORK_DIRECTORY, dataset_name, "logs", "removing_matches")
    Path(logs_path).mkdir(parents=True, exist_ok=True)
    matched_ngrams_dir = Path(WORK_DIRECTORY, dataset_name, "matched_ngrams") # it should already exist)
    output_dir = Path(WORK_DIRECTORY, dataset_name, "removed_data")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    

    logger.info(f"_____________")
    logger.info(f"ARGUMENTS")
    logger.info(f"- Logs directory: {logs_path}")
    logger.info(f"- Matched ngrams dir: {matched_ngrams_dir}")
    logger.info(f"- Shards jsonl: {args.shards_jsonl}")
    logger.info(f"- Node limit: {args.node_limit}")
    logger.info(f"- Account: {args.account}")
    logger.info(f"- Partition: {args.partition}")
    logger.info(f"- Time limit: {args.time_limit}")
    logger.info(f"- Output directory: {output_dir}")
    logger.info(f"- Number workers: {args.n_workers}")
    logger.info(f"- Match threshold: {args.match_threshold}")
    logger.info(f"- Memory: {args.memory}")
    logger.info(f"_____________")
    logger.info(f"Finding jobs to run...")

    if jobs_to_submit:
        logger.info(f"Found {len(jobs_to_submit)} jobs...")
    else:
        logger.info("No jobs to submit.")
        return
    
    logger.info(f"_____________")
    logger.info(f"Submitting jobs to the system...")


    # Set variables
    parse_jsonl = ""
    if args.parse_jsonl:
        parse_jsonl = "--parse-jsonl"
    batch_rows = [shards_dict[int(idx) - 1] for idx in jobs_to_submit]
    batch_jsonl_path = logs_path / f"batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    pl.DataFrame(batch_rows).write_ndjson(batch_jsonl_path)
    array_idxs = f"1-{len(jobs_to_submit)}"
    job_name = f"{dataset_name}_%A_%a"
    output_logs = f"{logs_path}/job_%A_%a.out"
    error_logs = f"{logs_path}/job_%A_%a.err"
    slurm_script_path = "utils/template_match_removal.slurm"


    cmd = ["sbatch", "--array", f"{array_idxs}",
            "--job-name", job_name,
            "--account", args.account,
            "--partition", args.partition,
            "--cpus-per-task", str(args.n_workers),
            "--mem", args.memory,
            "--time", args.time_limit,
            "--output", output_logs,
            "--error", error_logs,
            slurm_script_path,
            batch_jsonl_path,
            output_dir,
            str(args.n_workers),
            matched_ngrams_dir,
            str(args.match_threshold),
            parse_jsonl,
            TMP_DIR,
            DATASETS_DIR,
            metadata_path,
            ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"sbatch failed for job {job_name}: {e}")
        raise RuntimeError(f"sbatch failed: {e.stderr or e.stdout}") from e

    sbatch_job_id = result.stdout.strip().split()[-1]

    shards_dict_main = pl.read_ndjson(args.shards_jsonl).to_dicts()
    for seq_idx, actual_idx_str in enumerate(jobs_to_submit, start=1):
        job_id = f"{sbatch_job_id}_{seq_idx}"
        row = shards_dict_main[int(actual_idx_str) - 1]
        row["removal_job_id"] = job_id
        row["removal_status"] = "PENDING"
    pl.DataFrame(shards_dict_main).write_ndjson(args.shards_jsonl)

    logger.info(f"- Submitted...")


if __name__ == "__main__":
    main()