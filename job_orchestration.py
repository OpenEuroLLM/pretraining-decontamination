import argparse
import json
import yaml
from pathlib import Path
import os
import subprocess
import sys
import time
from collections import Counter

from loguru import logger

# Field in each jsonl line holding the job status.
STATUS_FIELD = "status"
# Jobs in these states occupy a queue slot.
ACTIVE_STATES = {"RUNNING", "PENDING"}
# Pool of jobs the submitter can still draw from.
SUBMITTABLE_STATE = "NOT_STARTED"

LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}"

with open('env_variables.yaml', 'r') as file:
    variables = yaml.safe_load(file)
    DECONTAMINATION_DIR = variables["DECONTAMINATION_DIR"]

def parse_args():
    p = argparse.ArgumentParser(
        description="Drip-feed SLURM array jobs, keeping the queue topped up.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # core 
    p.add_argument("--shards-jsonl", required=True,
                   help="Path to the shards jsonl (the source of truth for status).")
    p.add_argument("--max-queue", type=int, default=200,
                   help="Full queue size to keep filled (your --node-limit).")
    p.add_argument("--chunk", "-X", type=int, default=50,
                   help="How many NOT_STARTED jobs to submit per loop when there is room.")
    p.add_argument("--poll-interval", type=int, default=1800,
                   help="Seconds to sleep between status checks (default 30 min).")

    # passthrough submitter params (match your manual command) 
    p.add_argument("--account", required=True)
    p.add_argument("--partition", default="small")
    p.add_argument("--time-limit", default="6:00:00")
    p.add_argument("--n-workers", type=int, default=12)
    p.add_argument("--mem", default="448G")

    # removal phase
    p.add_argument("--removal-submitter", default="submitter_remove_matches_array.py",
                   help="Script used to submit removal jobs.")
    p.add_argument("--match-threshold", type=int, default=10,
                   help="Match threshold passed to the removal submitter.")
    p.add_argument("--skip-removal", action="store_true",
                   help="Exit after matching completes without running the removal phase.")

    # script locations / behaviour
    p.add_argument("--submitter", default="submitter_ngram_match_array.py")
    p.add_argument("--status-script", default="utils/status.py")
    p.add_argument("--python", default=sys.executable or "python3",
                   help="Python interpreter used to run the sub-scripts.")
    p.add_argument("--status-field", default=STATUS_FIELD,
                   help="Name of the status field in each jsonl line "
                        "(lets a future removal workflow use a different field).")
    p.add_argument("--max-cycles", type=int, default=0,
                   help="Stop after this many loop cycles (0 = run until nothing left to submit).")
    p.add_argument("--no-initial-status", action="store_true",
                   help="Skip the very first status.py update before the first fill.")

    # logging 
    p.add_argument("--workflow", default="matching",
                   help="Label for what this run does (e.g. 'matching', 'removal'); "
                        "used in the log file name.")
    p.add_argument("--log-dir", default="logs/job_orchestrator",
                   help="Directory where per-run log files are written.")
    return p.parse_args()


def count_statuses(path, status_field=STATUS_FIELD, missing_as=None):
    """Tally job statuses straight from the jsonl."""
    counts = Counter()
    total = 0
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"skipping unparseable jsonl line {ln}")
                continue
            raw = obj.get(status_field)
            status = str(missing_as if raw is None else raw).strip().upper()
            counts[status] += 1
            total += 1
    active = sum(counts.get(s, 0) for s in ACTIVE_STATES)
    not_started = counts.get(SUBMITTABLE_STATE, 0)
    done = total - active - not_started
    return {"counts": counts, "total": total,
            "active": active, "not_started": not_started, "done": done}


def log_snapshot(snap, phase="matching"):
    breakdown = ", ".join(f"{k}={v}" for k, v in sorted(snap["counts"].items())) or "(empty)"
    logger.info(f"[{phase}] jsonl: total={snap['total']} | active(RUNNING+PENDING)={snap['active']} | "
                f"NOT_STARTED={snap['not_started']} | done={snap['done']}")
    logger.info(f"[{phase}]        breakdown: {breakdown}")


def build_status_cmd(args):
    return [args.python, args.status_script, "--shards-jsonl", args.shards_jsonl]


def build_submit_cmd(args, node_limit):
    return [
        args.python, args.submitter,
        "--shards-jsonl", args.shards_jsonl,
        "--node-limit", str(node_limit),
        "--partition", args.partition,
        "--time-limit", args.time_limit,
        "--n-workers", str(args.n_workers),
        "--mem", args.mem,
        "--account", args.account,
    ]


def build_removal_submit_cmd(args, node_limit):
    return [
        args.python, args.removal_submitter,
        "--shards-jsonl", args.shards_jsonl,
        "--node-limit", str(node_limit),
        "--partition", args.partition,
        "--time-limit", args.time_limit,
        "--n-workers", str(args.n_workers),
        "--mem", args.mem,
        "--account", args.account,
        "--match-threshold", str(args.match_threshold),
    ]


def run_cmd(cmd):
    """Run a subprocess, streaming its output. Returns True on success."""
    logger.info(f"RUN: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"command failed (rc={e.returncode}): {' '.join(cmd)}")
        return False
    except FileNotFoundError as e:
        logger.error(f"command not found: {cmd[0]} ({e})")
        return False


def update_status(args):
    run_cmd(build_status_cmd(args))


def submit_n(args, n, reason):
    """Submit up to n NOT_STARTED jobs (submitter self-caps at available NOT_STARTED)."""
    if n <= 0:
        logger.info(f"no room to submit ({reason})")
        return
    logger.info(f"submitting up to {n} jobs ({reason})")
    run_cmd(build_submit_cmd(args, n))


def check_ngram_pkl(path):
    if not path.exists():
        logger.warning(f"required file not found, cannot submit removal jobs: {path}")
        sys.exit(1)


def sleep_until_next(seconds):
    wake = time.time() + seconds
    eest_wake = time.strftime("%H:%M:%S", time.localtime(wake))
    cest_wake = time.strftime("%H:%M:%S", time.localtime(wake - 3600))

    logger.info(f"sleeping {seconds / 60.0:.0f} min (next check -> CEST: ~{cest_wake} | EEST: ~{eest_wake})...")
    time.sleep(seconds)


def main():
    args = parse_args()

    # one log file per run: logs/job_orchestrator/orchestration_<workflow>_<dataset>_<timestamp>.log
    dataset = os.path.basename(os.path.dirname(os.path.abspath(args.shards_jsonl))) or "dataset"
    log_path = os.path.join(
        args.log_dir,
        f"orchestration_{args.workflow}_{dataset}_{time.strftime('%Y%m%d_%H%M%S')}.log",
    )
    logger.remove()
    logger.add(sys.stderr, format=LOG_FORMAT)
    logger.add(log_path, format=LOG_FORMAT)

    if not os.path.exists(args.shards_jsonl):
        logger.error(f"shards jsonl not found: {args.shards_jsonl}")
        sys.exit(1)
    if args.chunk > args.max_queue:
        logger.warning(f"--chunk ({args.chunk}) > --max-queue ({args.max_queue}); clamping to max-queue")
        args.chunk = args.max_queue
    if args.chunk <= 0:
        logger.error("--chunk must be > 0")
        sys.exit(1)

    logger.info("=== SLURM array-job orchestrator ===")
    logger.info(f"logging to {log_path}")
    logger.info(f"jsonl={args.shards_jsonl}  max-queue={args.max_queue}  chunk={args.chunk}  "
                f"poll={args.poll_interval}s")

    try:
        # 1. initial status update 
        if not args.no_initial_status:
            logger.info("initial status update ---")
            update_status(args)

        snap = count_statuses(args.shards_jsonl, args.status_field)
        log_snapshot(snap, phase="matching")

        if snap["not_started"] == 0 and snap["active"] == 0:
            logger.info("no NOT_STARTED or active matching jobs; skipping matching phase.")
        else:
            # 1.b first fill: top the queue up to max-queue
            if snap["not_started"] > 0:
                submit_n(args, max(0, args.max_queue - snap["active"]),
                         reason=f"first fill -> top up to {args.max_queue}")
            else:
                logger.info("no NOT_STARTED jobs to submit; monitoring active jobs until they finish")

            # 2. loop
            cycle = 0
            while True:
                cycle += 1
                if args.max_cycles and cycle > args.max_cycles:
                    logger.info(f"reached --max-cycles={args.max_cycles}; stopping")
                    break

                sleep_until_next(args.poll_interval)

                logger.info(f"cycle {cycle}: status update ---")
                update_status(args)
                snap = count_statuses(args.shards_jsonl, args.status_field)
                log_snapshot(snap, phase="matching")

                # stopping condition: nothing left to submit AND nothing still active
                if snap["not_started"] == 0 and snap["active"] == 0:
                    logger.info("all jobs finished (no NOT_STARTED, none active). done.")
                    break
                if snap["not_started"] == 0:
                    logger.info(f"no NOT_STARTED jobs left; waiting for {snap['active']} active to drain")
                    continue

                headroom = args.max_queue - snap["active"]
                submittable = min(args.chunk, snap["not_started"])
                to_submit = submittable if headroom >= submittable else 0
                reason = (f"headroom={headroom} >= submittable={submittable}"
                        if to_submit else
                        f"headroom={headroom} < submittable={submittable}, waiting")
                submit_n(args, to_submit, reason=reason)

        # ── Removal phase ────────────────────────────────────────────────────
        if args.skip_removal:
            logger.info("--skip-removal set; skipping removal phase.")
        else:
            logger.info("=== removal phase: checking prerequisites ===")
            update_status(args)
            with open(args.shards_jsonl) as f:
                all_rows = [json.loads(ln) for ln in f if ln.strip()]
            if not all(row.get("status") == "COMPLETED" for row in all_rows):
                logger.error(
                    "not all matching jobs have status=COMPLETED; "
                    "removal phase skipped — re-run after matching finishes."
                )
            else:
                logger.info("all matching jobs COMPLETED; starting removal phase.")
                REMOVAL_STATUS_FIELD = "removal_status"

                dataset_name = all_rows[0]["dataset_name"]
                language = Path(args.shards_jsonl).stem[len(dataset_name) + 1:]
                ngram_pkl = Path(DECONTAMINATION_DIR) / dataset_name / language / "matched_ngrams" / "all_matched_ngrams.pkl"
                logger.info(f"expecting ngram pkl at: {ngram_pkl}")

                snap = count_statuses(args.shards_jsonl, REMOVAL_STATUS_FIELD,
                                      missing_as=SUBMITTABLE_STATE)
                log_snapshot(snap, phase="removal")

                if snap["not_started"] == 0 and snap["active"] == 0:
                    logger.info("no removal jobs to submit; done.")
                else:
                    if snap["not_started"] > 0:
                        n = max(0, args.max_queue - snap["active"])
                        if n > 0:
                            check_ngram_pkl(ngram_pkl)
                            logger.info(f"submitting up to {n} removal jobs "
                                        f"(first fill -> top up to {args.max_queue})")
                            run_cmd(build_removal_submit_cmd(args, n))
                        else:
                            logger.info("no room to submit removal jobs (first fill)")
                    else:
                        logger.info("no NOT_STARTED removal jobs; monitoring active until they finish")

                    cycle = 0
                    while True:
                        cycle += 1
                        if args.max_cycles and cycle > args.max_cycles:
                            logger.info(f"reached --max-cycles={args.max_cycles}; stopping removal phase")
                            break

                        sleep_until_next(args.poll_interval)

                        logger.info(f"removal cycle {cycle}: status update ---")
                        update_status(args)
                        snap = count_statuses(args.shards_jsonl, REMOVAL_STATUS_FIELD,
                                              missing_as=SUBMITTABLE_STATE)
                        log_snapshot(snap, phase="removal")

                        if snap["not_started"] == 0 and snap["active"] == 0:
                            logger.info("all removal jobs finished (no NOT_STARTED, none active). done.")
                            break
                        if snap["not_started"] == 0:
                            logger.info(f"no NOT_STARTED removal jobs left; "
                                        f"waiting for {snap['active']} active to drain")
                            continue

                        headroom = args.max_queue - snap["active"]
                        submittable = min(args.chunk, snap["not_started"])
                        to_submit = submittable if headroom >= submittable else 0
                        reason = (f"headroom={headroom} >= submittable={submittable}"
                                  if to_submit else
                                  f"headroom={headroom} < submittable={submittable}, waiting")
                        if to_submit:
                            check_ngram_pkl(ngram_pkl)
                            logger.info(f"submitting up to {to_submit} removal jobs ({reason})")
                            run_cmd(build_removal_submit_cmd(args, to_submit))
                        else:
                            logger.info(f"no room to submit removal jobs ({reason})")

    except KeyboardInterrupt:
        logger.warning("interrupted by user. Running SLURM jobs are unaffected; re-run to resume.")
        sys.exit(130)

    logger.info("=== orchestrator finished ===")


if __name__ == "__main__":
    main()