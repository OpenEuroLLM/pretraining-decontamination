
import yaml
import zstandard as zstd
from pathlib import Path
from collections import defaultdict
import argparse
from loguru import logger

with open('env_variables.yaml', 'r') as file:
    variables = yaml.safe_load(file)
    WORK_DIRECTORY = variables["DECONTAMINATION_DIR"]
    Path(WORK_DIRECTORY).mkdir(exist_ok=True, parents=True)

parser = argparse.ArgumentParser(description="Combine per-worker removed JSONL files into a single file per shard.")
parser.add_argument("--dataset", required=True)
parser.add_argument("--lang", required=True)
parser.add_argument("--compress", action=argparse.BooleanOptionalAction, default=False,
                    help="Compress output as .jsonl.zst (default: plain .jsonl).")
parser.add_argument("--compression-level", type=int, default=9,
                    help="zstd compression level (1=fastest, 22=max). Only used with --compress. Default: 9.")
args = parser.parse_args()

INPUT_DIR  = f"{WORK_DIRECTORY}/{args.dataset}/{args.lang}/removed_data"
OUTPUT_DIR = f"{WORK_DIRECTORY}/{args.dataset}/{args.lang}/final_removed_data"
Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

logger.info("Dataset   : {}", args.dataset)
logger.info("Language  : {}", args.lang)
logger.info("Input     : {}", INPUT_DIR)
logger.info("Output    : {}", OUTPUT_DIR)
logger.info("Compress  : {}", args.compress)
if args.compress:
    logger.info("zstd level: {}", args.compression_level)

try:
    lang_prefix = f"{args.lang}_"
    out_lang_dir = Path(OUTPUT_DIR, args.lang)
    out_lang_dir.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for src_dir in Path(INPUT_DIR).iterdir():
        if not src_dir.is_dir():
            continue
        for f in src_dir.glob("*.jsonl"):
            stem = f.stem
            no_prefix = stem[len(lang_prefix):] if stem.startswith(lang_prefix) else stem
            # Strip UUID suffix: last _-separated token containing hyphens is a UUID
            parts = no_prefix.rsplit("_", 1)
            base = parts[0] if len(parts) == 2 and "-" in parts[1] else no_prefix
            groups[base].append(f)

    if not groups:
        logger.warning("No .jsonl files found under {}", INPUT_DIR)
    else:
        logger.info("Found {} file group(s) to combine", len(groups))

    if args.compress:
        cctx = zstd.ZstdCompressor(level=args.compression_level)
        for base, files in sorted(groups.items()):
            out_file = out_lang_dir / f"{base}.jsonl.zst"
            logger.info("Writing {} ({} part(s)) -> {}", base, len(files), out_file)
            with out_file.open("wb") as raw_out:
                with cctx.stream_writer(raw_out) as compressor:
                    for f in sorted(files):
                        with f.open("rb") as infile:
                            compressor.write(infile.read())
    else:
        for base, files in sorted(groups.items()):
            out_file = out_lang_dir / f"{base}.jsonl"
            logger.info("Writing {} ({} part(s)) -> {}", base, len(files), out_file)
            with out_file.open("w", encoding="utf-8") as outfile:
                for f in sorted(files):
                    with f.open("r", encoding="utf-8") as infile:
                        outfile.write(infile.read())

    logger.info("Done — {} group(s) written to {}", len(groups), out_lang_dir)

except Exception:
    logger.exception("combine_jsonl_files failed")
    raise SystemExit(1)